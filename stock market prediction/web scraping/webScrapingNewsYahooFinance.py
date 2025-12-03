from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from bs4 import BeautifulSoup
from urllib.parse import urljoin
from datetime import datetime
import pandas as pd
import time
import os

# ===========================
# Senarai ticker syarikat
# ===========================
tickers = ["AAPL", "AMD", "AMZN", "GOOG", "INTC", "META", "MSFT", "NFLX", "NVDA", "TSLA"]

# ===========================
# Setup Selenium (headless)
# ===========================
options = Options()
options.add_argument("--headless=new")  # run background
options.add_argument("--disable-gpu")
options.add_argument("--window-size=1920,1080")
options.add_argument("--no-sandbox")
options.add_argument("--disable-dev-shm-usage")
driver = webdriver.Chrome(options=options)

# ===========================
# Output directory
# ===========================
output_dir = r"C:\Users\AMJAD\PycharmProjects\fyp1test\Lib\News Stock Dataset"
os.makedirs(output_dir, exist_ok=True)

# ===========================
# Loop setiap ticker
# ===========================
for ticker in tickers:
    log_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    with open("log_scrape.txt", "a", encoding="utf-8") as log_file:
        log_file.write(f"Scraping {ticker} started at {log_time}\n")

    url = f"https://finance.yahoo.com/quote/{ticker}/news"
    driver.get(url)
    time.sleep(5)  # tunggu page load

    soup = BeautifulSoup(driver.page_source, "html.parser")

    # ===========================
    # Cari article list (multi selector)
    # ===========================
    articles = soup.find_all("li", attrs={"data-test-locator": "mega"})
    if not articles:
        articles = soup.find_all("li", class_="js-stream-content")
    if not articles:
        articles = soup.select("div.stream-item, li.stream-item")

    news_data = []
    for article in articles:
        try:
            title_tag = article.find("h3")
            link_tag = article.find("a")

            if title_tag and link_tag and link_tag.get("href"):
                title = title_tag.get_text(strip=True)
                link = urljoin("https://finance.yahoo.com", link_tag["href"])

                # ===========================
                # Masuk ke page artikel penuh
                # ===========================
                driver.get(link)
                time.sleep(3)  # bagi masa page load
                article_soup = BeautifulSoup(driver.page_source, "html.parser")

                content = ""

                # Cuba kes Yahoo asli
                article_main = article_soup.find("article")
                if article_main:
                    paragraphs = article_main.find_all("p")
                    content = " ".join(p.get_text(strip=True) for p in paragraphs if p.get_text(strip=True))

                # Kalau masih kosong, cuba fallback semua <p>
                if not content:
                    paragraphs = article_soup.find_all("p")
                    content = " ".join(p.get_text(strip=True) for p in paragraphs if p.get_text(strip=True))

                # Kalau masih fail → external site
                if not content or "Oops, something went wrong" in content:
                    canonical = article_soup.find("link", rel="canonical")
                    if canonical and canonical.get("href") and "yahoo.com" not in canonical["href"]:
                        ext_url = canonical["href"]
                        driver.get(ext_url)
                        time.sleep(3)
                        ext_soup = BeautifulSoup(driver.page_source, "html.parser")
                        paragraphs = ext_soup.find_all("p")
                        content = " ".join(p.get_text(strip=True) for p in paragraphs if p.get_text(strip=True))

                # Simpan hasil
                news_data.append({
                    "ticker": ticker,
                    "title": title,
                    "link": link,
                    "date": datetime.today().strftime("%Y-%m-%d"),
                    "content": content
                })

        except Exception as e:
            print(f"⚠️ Error parsing article for {ticker}: {e}")
            continue

    # Simpan CSV
    df = pd.DataFrame(news_data)
    today = datetime.now().strftime("%Y-%m-%d")
    filename = f"{ticker.lower()}_news_{today}.csv"
    file_path = os.path.join(output_dir, filename)
    df.to_csv(file_path, index=False, encoding="utf-8")

    # Papar ringkasan ke terminal
    print(f"\n📰 Berita untuk {ticker}: ({len(news_data)} artikel)")
    for news in news_data[:5]:  # preview max 5 artikel pertama
        print(f"- {news['title']}")
        print(f"  Content: {news['content'][:150]}...\n")
    print("=" * 80)

driver.quit()
print("\n🎉 Semua ticker siap scrape!")
