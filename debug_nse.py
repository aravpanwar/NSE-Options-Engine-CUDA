from nsepython import nse_optionchain_scrapper
import json

data = nse_optionchain_scrapper("NIFTY")
print(f"Type: {type(data)}")
print(f"Keys: {list(data.keys()) if isinstance(data, dict) else 'not a dict'}")
print(f"Preview:")
print(json.dumps(data, indent=2)[:1000])
