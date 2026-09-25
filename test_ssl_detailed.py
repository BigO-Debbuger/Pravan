"""
test_ssl_detailed.py
Diagnoses SSL connection to CDS endpoints without exposing credentials.
"""
import requests
import certifi
import ssl
import sys
import platform

print(f"Python: {sys.version}")
print(f"Platform: {platform.platform()}")
print(f"Certifi: {certifi.__version__}, store: {certifi.where()}")
print()

endpoints = [
    ("CDS Production", "https://cds.climate.copernicus.eu/api"),
]

print("--- Test 1: Standard requests (certifi CA bundle) ---")
for name, url in endpoints:
    try:
        r = requests.get(url, timeout=10)
        print(f"  {name}: OK (HTTP {r.status_code})")
    except Exception as e:
        print(f"  {name}: FAILED - {type(e).__name__}: {str(e)[:120]}")

print()
print("--- Test 2: truststore (Windows system CA store) ---")
try:
    import truststore
    truststore.inject_into_ssl()
    for name, url in endpoints:
        try:
            r = requests.get(url, timeout=10)
            print(f"  {name}: OK (HTTP {r.status_code})")
        except Exception as e:
            print(f"  {name}: FAILED - {type(e).__name__}: {str(e)[:120]}")
except ImportError:
    print("  truststore not installed")
except Exception as e:
    print(f"  truststore error: {e}")
