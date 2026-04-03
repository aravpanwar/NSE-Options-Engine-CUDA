from kiteconnect import KiteConnect
from dotenv import load_dotenv
import os
import webbrowser

load_dotenv()
api_key    = os.getenv("KITE_API_KEY")
api_secret = os.getenv("KITE_API_SECRET")

kite = KiteConnect(api_key=api_key)

# Open login URL in browser
login_url = kite.login_url()
print(f"Opening Zerodha login in browser...")
print(f"URL: {login_url}")
webbrowser.open(login_url)

# After login, Zerodha redirects to http://127.0.0.1:5000/?request_token=XXXX
# Copy the request_token from the URL bar and paste it here
request_token = input("\nPaste the request_token from the redirect URL: ").strip()

# Exchange for access token
data = kite.generate_session(request_token, api_secret=api_secret)
access_token = data["access_token"]

# Save to .env for reuse
with open(".env", "a") as f:
    f.write(f"\nKITE_ACCESS_TOKEN={access_token}")

print(f"\nAccess token saved to .env")
print(f"Token: {access_token}")
