import requests
from bs4 import BeautifulSoup

url = "https://fastapi.tiangolo.com/tutorial/"
html = requests.get(url).text

soup = BeautifulSoup(html, "html.parser")
text = soup.get_text()

with open("data/raw/fastapi_tutorial.txt", "w", encoding="utf-8") as f:
    f.write(text)
