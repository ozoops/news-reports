import os
import asyncio
from datetime import datetime
import json
from concurrent.futures import ThreadPoolExecutor
import shutil

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
PUBLISH_DIR = "."  # GitHub Pages가 읽는 루트에 최신본 복사

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
        <meta name="viewport" content="width=device-width, initial-scale=1" />
        <style>
            :root {{
                color-scheme: dark;
                font-family: 'Pretendard', 'Noto Sans KR', 'SUIT', system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
                --bg: #050505;
                --panel: #0f1117;
                --border: #1f2430;
                --text: #f5f7ff;
                --muted: #a0a7be;
                --accent: #6ea9ff;
            }}
            body {{
                margin: 0;
                min-height: 100vh;
                padding: 2.5rem 1.6rem 3rem;
                background: var(--bg);
                color: var(--text);
                line-height: 1.65;
                display: flex;
                justify-content: center;
            }}
            main {{
                width: min(960px, 100%);
                display: flex;
                flex-direction: column;
                gap: 1.5rem;
            }}
            h1 {{
                margin: 0;
                font-size: clamp(2rem, 5vw, 2.9rem);
            }}
            .table-wrap {{
                background: var(--panel);
                border-radius: 1.1rem;
                border: 1px solid var(--border);
                overflow: hidden;
                box-shadow: 0 25px 60px rgba(0,0,0,.5);
            }}
            table {{
                width: 100%;
                border-collapse: collapse;
            }}
            thead {{
                background: #131722;
            }}
            th {{
                padding: 1rem 1.2rem;
                text-align: left;
                font-weight: 600;
                letter-spacing: 0.02em;
                color: var(--muted);
            }}
            td {{
                padding: 1rem 1.2rem;
                border-top: 1px solid var(--border);
                vertical-align: top;
            }}
            tr:first-child td {{
                border-top: none;
            }}
            td.date {{
                width: 8rem;
                color: var(--muted);
                font-feature-settings: 'tnum';
                font-weight: 500;
            }}
            td.summary {{
                font-size: 1.05rem;
            }}
            td.summary p {{
                margin: 0.4rem 0 0;
                color: var(--muted);
            }}
            .keyword-pill {{
                display: inline-block;
                margin: 0 0.3rem 0.35rem 0;
                padding: 0.25rem 0.6rem;
                border-radius: 999px;
                background: rgba(110,169,255,0.12);
                color: var(--accent);
                font-size: 0.78rem;
                font-weight: 500;
            }}
            a.headline {{
                display: inline-block;
                color: var(--text);
                font-weight: 600;
                text-decoration: none;
                margin-top: 0.15rem;
            }}
            a.headline:hover {{ color: var(--accent); }}
            .source {{
                text-align: center;
                white-space: nowrap;
            }}
            a.source-pill {{
                display: inline-flex;
                align-items: center;
                justify-content: center;
                min-width: 5rem;
                padding: 0.45rem 0.9rem;
                border-radius: 999px;
                background: #1a1f2b;
                color: var(--muted);
                text-decoration: none;
                font-weight: 600;
                letter-spacing: 0.02em;
            }}
            a.source-pill:hover {{
                background: #252c3d;
                color: var(--text);
            }}
            @media (max-width: 680px) {{
                body {{ padding: 1.5rem 1.1rem 2.5rem; }}
                main {{ gap: 1rem; }}
                th, td {{ padding: 0.85rem; }}
                td.summary {{ font-size: 1rem; }}
                .table-wrap {{ border-radius: 0.9rem; box-shadow: none; }}
            }}
        </style>
    </head>
    <body>
        <main>
            <h1>{report_name} ({today_str})</h1>
            <div class="table-wrap">
                <table>
                    <thead>
                        <tr>
                            <th>날짜</th>
                            <th>제목 / 요약</th>
                            <th>출처</th>
                        </tr>
                    </thead>
                    <tbody>
    """

    for item in report_data:
        keywords = item.get("keywords", "")
        keyword_html = "".join(
            f'<span class="keyword-pill">{kw.strip()}</span>'
            for kw in keywords.split(",") if kw.strip()
        )
        source_label = item.get("source", "출처")
        summary_text = item.get("summary", "")

        html_content += f"""
                        <tr>
                            <td class="date">{item['date']}</td>
                            <td class="summary">
                                {keyword_html}
                                <a class="headline" href="{item['link']}" target="_blank" rel="noopener">{item['title']}</a>
                                <p>{summary_text}</p>
                            </td>
                            <td class="source">
                                <a class="source-pill" href="{item['link']}" target="_blank" rel="noopener">{source_label}</a>
                            </td>
                        </tr>
        """

    html_content += """
                    </tbody>
                </table>
            </div>
        </main>
    </body>
    </html>
    """

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(html_content)

    if PUBLISH_DIR:
        os.makedirs(PUBLISH_DIR, exist_ok=True)
        publish_path = os.path.join(PUBLISH_DIR, filename)
        if os.path.abspath(publish_path) != os.path.abspath(filepath):
            shutil.copyfile(filepath, publish_path)
            print(f"Published copy: {publish_path}")

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
