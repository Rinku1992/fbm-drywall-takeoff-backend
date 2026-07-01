"""
test_local.py — smoke test for the classifier service.

Usage examples:

  # Local test (after `uvicorn app.main:app --port 8080`):
  python test_local.py --project_id myproj --plan_id myplan --user_id alice

  # Cloud Run test:
  URL=$(gcloud run services describe drywall-page-classifier \\
          --region=us-central1 --format='value(status.url)')
  TOKEN=$(gcloud auth print-identity-token)
  python test_local.py --url $URL --token $TOKEN \\
      --project_id myproj --plan_id myplan --user_id alice
"""
import argparse
import time

import requests


def test_root(base_url, headers):
    print(f"\n─── GET / ───")
    r = requests.get(f"{base_url}/", headers=headers, timeout=10)
    print(f"  Status: {r.status_code}")
    print(f"  Body:   {r.json()}")
    return r.status_code == 200


def test_healthz(base_url, headers):
    print(f"\n─── GET /healthz ───")
    r = requests.get(f"{base_url}/healthz", headers=headers, timeout=30)
    print(f"  Status: {r.status_code}")
    print(f"  Body:   {r.json()}")
    return r.status_code == 200


def test_classify(base_url, headers, project_id, plan_id, user_id):
    print(f"\n─── POST /classify_pages ───")
    print(f"  project_id: {project_id}")
    print(f"  plan_id:    {plan_id}")
    print(f"  user_id:    {user_id}")

    payload = {
        "project_id": project_id,
        "plan_id": plan_id,
        "user_id": user_id,
    }

    h = dict(headers)
    h["Content-Type"] = "application/json"

    t0 = time.time()
    r = requests.post(f"{base_url}/classify_pages",
                      headers=h, json=payload, timeout=1800)
    elapsed = time.time() - t0

    print(f"  Status: {r.status_code} (elapsed: {elapsed:.2f}s)")
    if r.status_code == 200:
        body = r.json()
        print(f"  Inference time:  {body['inference_time_seconds']}s")
        print(f"  Total time:      {body['total_time_seconds']}s")
        print(f"  Pages returned:  {len(body['pages'])}")
        for p in body["pages"]:
            demoted = " [DEMOTED]" if p.get("threshold_demoted") else ""
            print(f"    page {p['page_number']:>3}: "
                  f"{p['plan_type']:<25} conf={p['confidence']:.3f}{demoted}")
    else:
        print(f"  Error body: {r.text}")
    return r.status_code == 200


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://localhost:8080")
    parser.add_argument("--token", default=None,
                        help="Bearer token (gcloud auth print-identity-token)")
    parser.add_argument("--project_id", required=False,
                        help="Project ID for the GCS path lookup.")
    parser.add_argument("--plan_id", required=False,
                        help="Plan ID for the GCS path lookup.")
    parser.add_argument("--user_id", required=False,
                        help="User ID; logged but not used for path resolution.")
    args = parser.parse_args()

    headers = {}
    if args.token:
        headers["Authorization"] = f"Bearer {args.token}"

    print(f"Testing {args.url}")

    ok = True
    ok &= test_root(args.url, headers)
    ok &= test_healthz(args.url, headers)

    if args.project_id and args.plan_id and args.user_id:
        ok &= test_classify(args.url, headers,
                            args.project_id, args.plan_id, args.user_id)
    else:
        print("\n⚠️  Skipping /classify_pages "
              "(need --project_id, --plan_id, and --user_id)")

    print(f"\n{'═' * 50}")
    print(f"{'ALL TESTS PASSED ✓' if ok else 'SOME TESTS FAILED ✗'}")
    print(f"{'═' * 50}")


if __name__ == "__main__":
    main()
