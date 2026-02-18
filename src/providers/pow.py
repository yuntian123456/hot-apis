import wasmtime
import json
import base64
from pathlib import Path
from typing import Dict, Optional
import ctypes

_wasm_store = None
_wasm_instance = None

def _init_wasm():
    global _wasm_store, _wasm_instance
    
    if _wasm_instance is not None:
        return _wasm_store, _wasm_instance
    
    wasm_path = Path(__file__).parent.parent.parent / "sha3_wasm_bg.wasm"
    if not wasm_path.exists():
        raise FileNotFoundError(f"WASM file not found: {wasm_path}")
    
    engine = wasmtime.Engine()
    store = wasmtime.Store(engine)
    module = wasmtime.Module.from_file(engine, str(wasm_path))
    
    instance = wasmtime.Instance(store, module, [])
    
    _wasm_store = store
    _wasm_instance = instance
    
    return store, instance

def solve_pow_wasm(challenge: str, salt: str, difficulty: int, expire_at: int) -> Optional[int]:
    try:
        store, instance = _init_wasm()
        
        exports = instance.exports(store)
        memory = exports["memory"]
        wasm_solve = exports["wasm_solve"]
        add_to_stack = exports["__wbindgen_add_to_stack_pointer"]
        alloc = exports["__wbindgen_export_0"]
        realloc = exports["__wbindgen_export_1"]
        
        prefix = f"{salt}_{expire_at}_"
        
        challenge_bytes = challenge.encode('utf-8')
        prefix_bytes = prefix.encode('utf-8')
        
        challenge_len = len(challenge_bytes)
        prefix_len = len(prefix_bytes)
        
        challenge_ptr = alloc(store, challenge_len, 1)
        prefix_ptr = alloc(store, prefix_len, 1)
        
        memory.write(store, challenge_bytes, challenge_ptr)
        memory.write(store, prefix_bytes, prefix_ptr)
        
        retptr = add_to_stack(store, -16)
        
        wasm_solve(store, retptr, challenge_ptr, challenge_len, prefix_ptr, prefix_len, float(difficulty))
        
        result_bytes = memory.read(store, retptr, retptr + 16)
        status = int.from_bytes(result_bytes[:4], 'little')
        import struct
        value = int(struct.unpack('<d', result_bytes[8:16])[0])
        
        add_to_stack(store, 16)
        
        if status == 0:
            return None
        
        return value
        
    except Exception as e:
        print(f"WASM PoW error: {e}")
        import traceback
        traceback.print_exc()
        return None

def solve_pow_fallback(challenge: str, salt: str, difficulty: int, expire_at: int) -> Optional[int]:
    import hashlib
    
    def keccak256(data: bytes) -> bytes:
        k = hashlib.sha3_256()
        k.update(data)
        return k.digest()
    
    prefix = f"{salt}_{expire_at}_"
    prefix_bytes = prefix.encode('utf-8')
    challenge_bytes = challenge.encode('utf-8')
    
    target_prefix = b'\x00' * (difficulty // 8)
    remaining_bits = difficulty % 8
    
    nonce = 0
    max_iterations = 10000000
    
    while nonce < max_iterations:
        nonce_bytes = str(nonce).encode('utf-8')
        data = challenge_bytes + prefix_bytes + nonce_bytes
        
        hash_result = keccak256(data)
        
        if difficulty <= 8:
            if hash_result[0] >> (8 - difficulty) == 0:
                return nonce
        else:
            prefix_match = hash_result[:difficulty // 8] == target_prefix
            if prefix_match:
                if remaining_bits == 0:
                    return nonce
                elif (hash_result[difficulty // 8] >> (8 - remaining_bits)) == 0:
                    return nonce
        
        nonce += 1
    
    return None

def solve_pow_challenge(challenge: Dict) -> str:
    algorithm = challenge["algorithm"]
    challenge_str = challenge["challenge"]
    salt = challenge["salt"]
    difficulty = challenge["difficulty"]
    expire_at = challenge["expire_at"]
    signature = challenge["signature"]
    target_path = challenge["target_path"]
    
    if algorithm != "DeepSeekHashV1":
        raise ValueError(f"Unsupported algorithm: {algorithm}")
    
    answer = solve_pow_wasm(challenge_str, salt, difficulty, expire_at)
    
    if answer is None:
        answer = solve_pow_fallback(challenge_str, salt, difficulty, expire_at)
    
    if answer is None:
        raise RuntimeError("Failed to solve PoW challenge")
    
    pow_response = {
        "algorithm": algorithm,
        "challenge": challenge_str,
        "salt": salt,
        "answer": answer,
        "signature": signature,
        "target_path": target_path
    }
    
    return base64.b64encode(json.dumps(pow_response).encode()).decode()

if __name__ == "__main__":
    import sys
    sys.path.insert(0, '.')
    
    test_challenge = {
        "algorithm": "DeepSeekHashV1",
        "challenge": "94e422f2ac55677000b92e561bb1a10da1a7fad54af93fa4706e4c1fa06eba5c",
        "salt": "9fa6d396e71f769c77ee",
        "difficulty": 144000,
        "expire_at": 1771229508176,
        "signature": "5c3b59d95f2810681e60601850583b347bad470734f6aa22bb5b5ab55aa50271",
        "target_path": "/api/v0/chat/completion"
    }
    
    result = solve_pow_challenge(test_challenge)
    print(f"Result: {result[:100]}...")
