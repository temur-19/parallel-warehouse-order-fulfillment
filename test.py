import dotenv
import os

dotenv.load_dotenv()

print({"BOT_TOKEN":os.getenv("TELEGRAM_BOT_TOKEN"),
       "ADMIN_ID":os.getenv("TELEGRAM_ADMIN_ID")})