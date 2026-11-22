"""Unit-test 01's S3 listing parser (pagination, token encoding) and MD5/ETag-verified download with mocked HTTP."""
import matplotlib; matplotlib.use("Agg")
import sys; sys.path.insert(0, ".")
from build import code_cells
import hashlib, types, io, contextlib
from pathlib import Path
ns = {}
with contextlib.redirect_stdout(io.StringIO()):
    for c in code_cells("src/01_dataset_acquisition.py")[:4]:
        exec(c, ns)
good = hashlib.md5(b"hello").hexdigest()
page1 = f'''<?xml version="1.0"?><ListBucketResult xmlns="http://s3.amazonaws.com/doc/2006-03-01/">
<IsTruncated>true</IsTruncated><NextContinuationToken>tok+1/=</NextContinuationToken>
<Contents><Key>challenge-2019/1.0.0/training/training_setA/p000001.psv</Key><ETag>"{good}"</ETag><Size>5</Size></Contents>
</ListBucketResult>'''.encode()
page2 = b'''<?xml version="1.0"?><ListBucketResult xmlns="http://s3.amazonaws.com/doc/2006-03-01/">
<IsTruncated>false</IsTruncated>
<Contents><Key>challenge-2019/1.0.0/training/training_setA/p000002.psv</Key><ETag>"bad"</ETag><Size>5</Size></Contents>
<Contents><Key>challenge-2019/1.0.0/training/training_setA/</Key><ETag>"x"</ETag><Size>0</Size></Contents>
</ListBucketResult>'''
calls = []
def fake_get(url, stream=False):
    calls.append(url)
    if "list-type" in url:
        return types.SimpleNamespace(content=page2 if "continuation-token" in url else page1)
    return types.SimpleNamespace(content=b"hello")
ns["http_get"] = fake_get
lst = ns["list_s3"]("challenge-2019/1.0.0/training/training_setA/")
assert [x["key"].split("/")[-1] for x in lst] == ["p000001.psv", "p000002.psv"], lst
assert "continuation-token=tok%2B1/%3D" in calls[1], calls[1]
d = Path("/tmp/dltest"); d.mkdir(exist_ok=True)
for f in d.glob("*"): f.unlink()
assert ns["download_one"]("u", d / "p1.psv", 5, lst[0]["etag"]) == "downloaded"
assert ns["download_one"]("u", d / "p1.psv", 5, lst[0]["etag"]) == "skipped"
try:
    ns["download_one"]("u", d / "p2.psv", 5, "bad"); raise SystemExit("bad ETag was accepted!")
except IOError as e:
    print("bad ETag correctly rejected:", e)
assert not (d / "p2.psv").exists()
print("PASS: S3 listing/pagination + MD5-ETag verified download")
