import os
import asyncio
from datetime import datetime
import json
from concurrent.futures import ThreadPoolExecutor

import feedparser
from newspaper import Article, ArticleException
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import nltk

# --- Configuration ---
RSS_FEEDS = [
    "https://news.google.com/rss?hl=ko&gl=KR&ceid=KR:ko",
    "https://www.yonhapnewstv.co.kr/browse/feed/",
    "https://www.hankyung.com/feed/all-news",
    "https://www.mk.co.kr/rss/40300009/",
    "https://biz.chosun.com/rss-feed/",
    "https://rss.donga.com/total.xml",
    "https://www.seoul.co.kr/rss/economy.xml",
    "https://www.newsfarm.co.kr/rss/allArticle.xml",
    "https://www.agriculture.co.kr/rss/rss.xml",
    "https://www.yna.co.kr/rss/politics-economy.xml"
]

REPORTS_CONFIG = [
    {
        "name": "농업_농협_리포트",
        "keywords": ["농협중앙회", "농협", "농촌", "농업", "축산업"]
    },
    {
        "name": "경제_리포트",
        "keywords": ["경제", "금리", "환율", "증시", "부동산", "물가", "수출", "무역", "투자"]
    }
]

MAX_ARTICLES_PER_FEED = 20  # Increase the number of articles to get better coverage
MAX_CONCURRENT_WORKERS = 10
OUTPUT_DIR = "reports"

# --- Article Fetching and Parsing (Adapted from backend/main.py) ---

def _fetch_and_parse_article(entry, source_name):
    """(Sync) Fetches and parses a full article from a URL."""
    link = entry.get("link")
    if not link:
        return None
    try:
        article = Article(link, language='ko')
        article.download()
        article.parse()
        
        if not article.text:
            return None

        return {
            "date": (article.publish_date or datetime.now()).strftime("%Y-%m-%d"),
            "title": article.title or entry.get("title", "제목 없음"),
            "source": source_name,
            "body": article.text,
            "link": link
        }
    except Exception as e:
        print(f"  - Error processing article at {link}: {e}")
        return None

async def _fetch_and_parse_all_articles():
    """Fetches all entries from RSS_FEEDS and parses them concurrently."""
    all_entries = []
    for feed_url in RSS_FEEDS:
        try:
            print(f"Fetching feed list: {feed_url}")
            feed = feedparser.parse(feed_url)
            source_name = feed.feed.get("title", "Unknown Source")
            for i, entry in enumerate(feed.entries):
                if i >= MAX_ARTICLES_PER_FEED: break
                all_entries.append((entry, source_name))
        except Exception as e:
            print(f"Error fetching or parsing feed {feed_url}: {e}")

    print(f"Found {len(all_entries)} total articles to process. Fetching concurrently...")
    loop = asyncio.get_running_loop()
    with ThreadPoolExecutor(max_workers=MAX_CONCURRENT_WORKERS) as executor:
        tasks = [
            loop.run_in_executor(executor, _fetch_and_parse_article, entry, source)
            for entry, source in all_entries
        ]
        results = await asyncio.gather(*tasks)
    
    parsed_articles = [item for item in results if item]
    print(f"Successfully parsed {len(parsed_articles)} articles.")
    return parsed_articles

# --- AI Summarization ---

def summarize_article_with_llm(article_body: str, llm: ChatOpenAI):
    """Summarizes the article body using an LLM."""
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an expert summarizer. Please provide a one-sentence summary of the following news article text."),
        ("human", "{text}")
    ])
    chain = prompt | llm | StrOutputParser()
    summary = chain.invoke({"text": article_body})
    return summary

# --- Report Generation ---

def generate_html_report(report_data: list, report_name: str):
    """Generates an HTML report from the summarized data."""
    today_str = datetime.now().strftime("%Y-%m-%d")
    filename = f"{report_name}_{today_str}.html"
    filepath = os.path.join(OUTPUT_DIR, filename)

    html_content = f"""
    <html>
    <head>
        <title>{report_name} - {today_str}</title>
        <meta charset="UTF-8">
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            h1 {{ color: #333; }}
            table {{ width: 100%; border-collapse: collapse; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            th {{ background-color: #f2f2f2; }}
            tr:nth-child(even) {{ background-color: #f9f9f9; }}
            a {{ color: #0066cc; text-decoration: none; }}
            a:hover {{ text-decoration: underline; }}
        </style>
    </head>
    <body>
        <h1>{report_name} ({today_str})</h1>
        <table>
            <thead>
                <tr>
                    <th>날짜</th>
                    <th>주요 키워드</th>
                    <th>제목 / 요약</th>
                    <th>출처</th>
                </tr>
            </thead>
            <tbody>
    """

    for item in report_data:
        html_content += f"""
                <tr>
                    <td>{item['date']}</td>
                    <td>{item['keywords']}</td>
                    <td>
                        <strong><a href="{item['link']}" target="_blank">{item['title']}</a></strong>
                        <br>
                        {item['summary']}
                    </td>
                    <td>{item['source']}</td>
                </tr>
        """

    html_content += """
            </tbody>
        </table>
    </body>
    </html>
    """

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(html_content)
    
    print(f"Successfully generated report: {filepath}")

# --- Main Execution ---

async def main():
    """Main function to generate daily news reports."""
    print("Starting daily report generation...")
    load_dotenv()
    
    # Ensure NLTK data is available
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        print("Downloading NLTK 'punkt' tokenizer data...")
        nltk.download('punkt')

    if not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY is not set. Please set it in your .env file.")
        return

    llm = ChatOpenAI(temperature=0, model="gpt-4o-mini")
    
    all_articles = await _fetch_and_parse_all_articles()

    for config in REPORTS_CONFIG:
        report_name = config["name"]
        keywords = config["keywords"]
        print(f"\n--- Generating report for: {report_name} ---")
        
        # Filter articles based on keywords
        filtered_articles = [
            article for article in all_articles
            if any(keyword.lower() in (article['title'] + article['body']).lower() for keyword in keywords)
        ]
        
        print(f"Found {len(filtered_articles)} articles for this report.")
        
        report_data = []
        for article in filtered_articles:
            print(f"  - Summarizing: {article['title']}")
            try:
                summary = summarize_article_with_llm(article['body'], llm)
                report_data.append({
                    "date": article["date"],
                    "keywords": ", ".join([kw for kw in keywords if kw.lower() in (article['title'] + article['body']).lower()]),
                    "title": article["title"],
                    "summary": summary,
                    "link": article["link"],
                    "source": article["source"]
                })
            except Exception as e:
                print(f"    - Failed to summarize article: {e}")
        
        if report_data:
            generate_html_report(report_data, report_name)
        else:
            print("No data to generate report.")

    print("\nAll reports generated successfully.")

if __name__ == "__main__":
    asyncio.run(main())
