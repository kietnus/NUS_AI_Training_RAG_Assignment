"""Upload the sample documents (sample_docs/manifest.json) through the API.

Usage (backend must be running):
    python scripts/seed_sample_docs.py            # upload, skip versions that already exist
    python scripts/seed_sample_docs.py --reset    # first delete ALL documents of the sample tenants
"""

import argparse

from common import DEFAULT_BASE_URL, SAMPLE_DIR, Api, load_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--reset", action="store_true",
                        help="delete every document of the tenants in the manifest before uploading")
    args = parser.parse_args()

    api = Api(args.base_url)
    manifest = load_manifest()

    if args.reset:
        for tenant in sorted({m["tenant"] for m in manifest}):
            for doc in api.documents(tenant):
                api.delete(tenant, doc["id"])
                print(f"deleted  [{tenant}] {doc['title']} v{doc['version']}")

    # Manifest order matters: v1.0 is uploaded before v2.0, so v2.0 expires v1.0.
    for item in manifest:
        path = SAMPLE_DIR / item["file"]
        metadata = {k: item[k] for k in ("title", "version", "valid_from", "valid_to")}
        response = api.upload(item["tenant"], path, metadata)
        if response.status_code == 201:
            r = response.json()
            extra = ""
            if r["expired_documents"]:
                extra = " | expired: " + ", ".join(f"v{x['version']}" for x in r["expired_documents"])
            print(f"uploaded [{item['tenant']}] {item['title']} v{item['version']}: "
                  f"{r['chunks_created']['semantic']} semantic / {r['chunks_created']['structure']} "
                  f"structure chunks ({r['ingestion_ms']} ms){extra}")
        elif response.status_code == 409:
            print(f"skipped  [{item['tenant']}] {item['title']} v{item['version']}: already uploaded")
        else:
            raise SystemExit(f"FAILED   {item['file']}: {response.status_code} {response.text}")

    print("\nDocuments per tenant:")
    for tenant in sorted({m["tenant"] for m in manifest}):
        for doc in api.documents(tenant):
            print(f"  [{tenant}] {doc['title']} v{doc['version']}: {doc['effective_status']}")


if __name__ == "__main__":
    main()
