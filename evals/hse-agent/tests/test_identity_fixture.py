import json
from pathlib import Path


EVAL_ROOT = Path(__file__).resolve().parents[1]
IDENTITIES = EVAL_ROOT / "fixtures" / "identities" / "users.json"
BASE_SEED = EVAL_ROOT / "fixtures" / "database" / "base_seed.sql"
KNOWN_ADMIN123_HASH = "$2a$10$7JB720yubVSZvUI0rEqK/.VqGOZTH.ulu33dHOiBE8ByOhJIrdAu2"


def test_identity_passwords_match_seed_hash_contract():
    """验证 identity passwords match seed hash contract 场景。"""
    identities = json.loads(IDENTITIES.read_text(encoding="utf-8"))
    seed = BASE_SEED.read_text(encoding="utf-8")

    assert identities["system_admin"]["username"] == "admin"
    assert identities["system_admin"]["user_id"] == 1
    assert all(account["password"] == "admin123" for account in identities.values())
    assert KNOWN_ADMIN123_HASH in seed
