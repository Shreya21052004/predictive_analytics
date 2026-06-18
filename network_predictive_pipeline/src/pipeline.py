import argparse
from collections import Counter
import json
from pathlib import Path

from .config import PipelineConfig, TARGET_CATEGORIES
from .metric_registry import REGISTRY
from .models import load_models, train_models
from .mongo_io import load_documents, write_predictions
from .normalizer import normalize_category, normalize_documents, normalize_provider
from .prediction import generate_predictions
from .validation import validate_normalized


def make_config(args):
    return PipelineConfig(
        mongo_uri=args.mongo_uri,
        db_name=args.db,
        source_collection=args.source,
        output_collection=args.out,
        models_dir=Path(args.models_dir),
        limit=args.limit,
    )


def load_normalized(config):
    docs = load_documents(
        config.mongo_uri,
        config.db_name,
        config.source_collection,
        limit=config.limit,
        categories=TARGET_CATEGORIES,
    )
    df = normalize_documents(docs)
    limit_msg = "all documents" if config.limit == 0 else f"up to {config.limit} documents"
    print(
        f"Loaded {len(docs)} Mongo document(s), normalized to {len(df)} resource-time row(s) "
        f"from {limit_msg}. Categories: {', '.join(TARGET_CATEGORIES)}."
    )
    return df


def cmd_validate(config):
    df = load_normalized(config)
    issues = validate_normalized(df)
    if issues:
        print("Validation issues:")
        for issue in issues:
            print(f"- {issue}")
        return 1
    print(f"Validation passed for {len(df)} normalized resource-time rows.")
    return 0


def metric_is_mapped(category, provider, metric):
    for provider_map in REGISTRY.get(category, {}).values():
        if metric in provider_map.get(provider, {}):
            return True
    return False


def cmd_inspect(config):
    docs = load_documents(
        config.mongo_uri,
        config.db_name,
        config.source_collection,
        limit=config.limit,
        categories=TARGET_CATEGORIES,
    )
    raw = Counter()
    mapped = Counter()
    unmapped = Counter()
    for doc in docs:
        category = normalize_category(doc.get("category"))
        provider = normalize_provider(doc.get("service_name") or doc.get("provider"))
        metric = doc.get("metric")
        if not category or not provider or not metric:
            continue
        key = (category, provider, metric)
        raw[key] += 1
        if metric_is_mapped(category, provider, metric):
            mapped[key] += 1
        else:
            unmapped[key] += 1

    df = normalize_documents(docs)
    available_cols = [col for col in df.columns if col.endswith("_available")] if not df.empty else []
    total_slots = len(df) * len(available_cols)
    available_slots = int(df[available_cols].astype(bool).to_numpy().sum()) if available_cols else 0
    mapped_docs = sum(mapped.values())
    unmapped_docs = sum(unmapped.values())
    raw_metric_docs = mapped_docs + unmapped_docs
    mapping_pct = (mapped_docs / raw_metric_docs * 100) if raw_metric_docs else 0
    density_pct = (available_slots / total_slots * 100) if total_slots else 0

    print(f"Inspected {len(docs)} Mongo document(s).")
    print(f"Normalized rows: {len(df)}")
    print(f"Raw metric mapping coverage: {mapped_docs}/{raw_metric_docs} ({mapping_pct:.2f}%)")
    print(f"Canonical feature density: {available_slots}/{total_slots} ({density_pct:.2f}%)")
    print("\nTop mapped raw metrics:")
    for (category, provider, metric), count in mapped.most_common(25):
        print(f"- {category} / {provider} / {metric}: {count}")
    print("\nTop unmapped raw metrics:")
    for (category, provider, metric), count in unmapped.most_common(50):
        print(f"- {category} / {provider} / {metric}: {count}")
    return 0


def cmd_train(config):
    df = load_normalized(config)
    issues = validate_normalized(df)
    if issues:
        print("Validation issues found. Training stopped:")
        for issue in issues:
            print(f"- {issue}")
        return 1
    trained = train_models(df, config.models_dir)
    print(f"Trained {len(trained)} category model(s): {', '.join(trained.keys()) or 'none'}")
    for category, artifact in trained.items():
        eval_metrics = artifact.get("eval_metrics")
        if eval_metrics:
            kind = artifact.get("kind", "unknown")
            print(f"  [{category}] kind={kind} eval_metrics={json.dumps(eval_metrics)}")
    return 0


def cmd_predict(config, dry_run=False):
    df = load_normalized(config)
    return predict_from_df(config, df, dry_run=dry_run)


def predict_from_df(config, df, dry_run=False):
    models = load_models(config.models_dir)
    if not models:
        print("No model artifacts found. Training models first.")
        models = train_models(df, config.models_dir)
    predictions = generate_predictions(df, models, target_categories=TARGET_CATEGORIES)

    if dry_run:
        sample = predictions[:10]
        print(json.dumps(sample, indent=2, default=str))
        print(f"Dry run produced {len(predictions)} Network prediction(s).")
        return 0

    if not predictions:
        print("No predictions generated.")
        return 0

    inserted = write_predictions(config.mongo_uri, config.db_name, config.output_collection, predictions)
    print(f"Wrote {inserted} Network prediction(s) → {config.db_name}.{config.output_collection}.")
    return 0


def cmd_run(config, dry_run=False):
    df = load_normalized(config)
    issues = validate_normalized(df)
    if issues:
        print("Validation issues:")
        for issue in issues:
            print(f"- {issue}")
        return 1
    print(f"Validation passed for {len(df)} normalized resource-time rows.")
    trained = train_models(df, config.models_dir)
    print(f"Trained {len(trained)} category model(s): {', '.join(trained.keys()) or 'none'}")
    return predict_from_df(config, df, dry_run=dry_run)


def parser():
    p = argparse.ArgumentParser(description="Network predictive analytics pipeline (standalone)")
    p.add_argument("command", choices=["inspect", "validate", "train", "predict", "run"])
    p.add_argument("--mongo-uri", default="mongodb://localhost:27017")
    p.add_argument("--db", default="mydb")
    p.add_argument("--source", default="network_data")
    p.add_argument("--out", default="22prediction_logs_network")
    p.add_argument("--models-dir", default="models")
    p.add_argument("--limit", type=int, default=0, help="Max Mongo documents to load. Use 0 for all documents.")
    p.add_argument("--dry-run", action="store_true")
    return p


def main():
    args = parser().parse_args()
    config = make_config(args)
    if args.command == "validate":
        raise SystemExit(cmd_validate(config))
    if args.command == "inspect":
        raise SystemExit(cmd_inspect(config))
    if args.command == "train":
        raise SystemExit(cmd_train(config))
    if args.command == "predict":
        raise SystemExit(cmd_predict(config, dry_run=args.dry_run))
    if args.command == "run":
        raise SystemExit(cmd_run(config, dry_run=args.dry_run))


if __name__ == "__main__":
    main()
