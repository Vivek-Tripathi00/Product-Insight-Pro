import os
import re
import requests
import joblib
import matplotlib.pyplot as plt
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from bs4 import BeautifulSoup
from flask import Flask, render_template, request, jsonify, session
from wordcloud import WordCloud
from datetime import datetime
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import IsolationForest
import time
from functools import lru_cache
from flask import Flask, render_template, request, flash, redirect, url_for
from flask_mail import Mail, Message

nltk.download("vader_lexicon")
nltk.download('stopwords')

app = Flask(__name__)
app.secret_key = 'your_secret_key_here'  # For session management

# for email newslatter
app.config['MAIL_SERVER'] = 'smtp.gmail.com'
app.config['MAIL_PORT'] = 587
app.config['MAIL_USERNAME'] = os.getenv('MAIL_USERNAME')  # Your email
app.config['MAIL_PASSWORD'] = os.getenv('MAIL_PASSWORD')  # Your email password or app password
app.config['MAIL_DEFAULT_SENDER'] = os.getenv('MAIL_USERNAME')
app.config['MAIL_USE_TLS'] = True
app.config['MAIL_USE_SSL'] = False

mail = Mail(app)

@app.route('/')
def subscribe():
    return render_template('index.html')

@app.route('/subscribe', methods=['POST'])
def handle_subscription():
    email = request.form['email']
    try:
        msg = Message('Newsletter Subscription',
                      recipients=[email])
        msg.html = """
<!DOCTYPE html>
<html>
<head>
  <style>
    body {
      font-family: 'Arial', sans-serif;
      background-color: #f9f9f9;
      padding: 0;
      margin: 0;
    }
    .container {
      max-width: 600px;
      margin: 30px auto;
      background: white;
      border-radius: 10px;
      padding: 30px;
      box-shadow: 0 5px 20px rgba(0,0,0,0.1);
    }
    h1 {
      color: #333333;
      font-size: 28px;
      text-align: center;
    }
    p {
      font-size: 16px;
      line-height: 1.6;
      color: #555555;
    }
    .header-image {
      text-align: center;
      margin-bottom: 20px;
    }
    .cta {
      display: inline-block;
      margin-top: 20px;
      padding: 12px 24px;
      background-color: #4CAF50;
      color: white;
      text-decoration: none;
      border-radius: 5px;
      font-weight: bold;
    }
    .footer {
      font-size: 12px;
      color: #aaa;
      text-align: center;
      margin-top: 30px;
    }
  </style>
</head>
<body>
  <div class="container">
    <div class="header-image">
      <img src="https://t4.ftcdn.net/jpg/13/97/95/65/240_F_1397956586_xIuJjfNgn6CdVxbZak7OoUL553rIEvX4.jpg" width="120" alt="Email icon" />
    </div>
    <h1>🎉 Welcome to Our Newsletter!</h1>
    <p>Hi there! 👋</p>
    <p>
      We're thrilled to have you on board. You've just joined a vibrant community that loves to stay updated, learn, and grow! 💡
    </p>
    <p>
      Here's what you can expect from us:
      <ul>
        <li>📬 Fresh updates right in your inbox</li>
        <li>🔥 Latest trends and insights</li>
        <li>🎁 Exclusive offers & giveaways</li>
      </ul>
    </p>
    <p>
      Stay tuned for your first newsletter. Until then, follow us on social media for more fun updates!
    </p>
    <div style="text-align:center;">
      <a class="cta" href="https://yourwebsite.com" target="_blank">Visit Our Website 🌐</a>
    </div>
    <div class="footer">
      <p>You received this email because you subscribed to our newsletter.</p>
      <p>Unsubscribe at any time. No hard feelings 💔</p>
    </div>
  </div>
</body>
</html>
"""

        mail.send(msg)
        flash('Subscription successful! A confirmation email has been sent.', 'success')
    except Exception as e:
        flash(f'An error occurred: {str(e)}', 'danger')
    return redirect(url_for('subscribe'))



# Cache frequently accessed data
@lru_cache(maxsize=128)
def get_sentiment_analyzer():
    return SentimentIntensityAnalyzer()

# Enhanced scraping with retries and better headers
def get_page_content(url, platform):
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
        "Accept-Language": "en-US,en;q=0.9",
        "Accept-Encoding": "gzip, deflate, br",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8",
        "Referer": "https://www.google.com/"
    }
    
    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            
            # Platform-specific content checks
            if platform == 'amazon' and "Sorry, we just need to make sure you're not a robot" in response.text:
                raise Exception("Amazon bot detection triggered")
                
            return response.text
        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {str(e)}")
            if attempt == max_retries - 1:
                raise
            time.sleep(2 ** attempt)  # Exponential backoff
    return None

# Enhanced Flipkart scraper
def scrape_flipkart_reviews(url, num_reviews=50):
    reviews = []
    page = 1

    while len(reviews) < num_reviews:
        page_url = f"{url}&page={page}"
        print(f"Scraping: {page_url}")
        
        try:
            html_content = get_page_content(page_url, 'flipkart')
            if not html_content:
                break

            soup = BeautifulSoup(html_content, "html.parser")
            review_blocks = soup.find_all("div", class_="cPHDOP col-12-12")

            if not review_blocks:
                print("No reviews found on this page.")
                break

            for review_block in review_blocks:
                try:
                    # Extract rating
                    rating_tag = review_block.find("div", class_="XQDdHH Ga3i8K")
                    rating_text = rating_tag.get_text(strip=True) if rating_tag else "0"
                    rating = int(float(re.search(r'\d+', rating_text).group())) if re.search(r'\d+', rating_text) else 0

                    # Extract other elements
                    title_tag = review_block.find("p", class_="z9E0IG")
                    title = title_tag.get_text(strip=True) if title_tag else "No Title"

                    review_text_tag = review_block.find("div", class_="ZmyHeo").find("div", class_="")
                    review_text = review_text_tag.get_text(strip=True) if review_text_tag else "No Review Text"

                    name_tag = review_block.find("p", class_="_2NsDsF AwS1CA")
                    reviewer_name = name_tag.get_text(strip=True) if name_tag else "Anonymous"

                    date_location_tag = review_block.find("p", class_="MztJPv")
                    date_location = date_location_tag.get_text(strip=True) if date_location_tag else "Unknown"

                    date_tag = date_location_tag.find_next_sibling("p")
                    date = date_tag.get_text(strip=True) if date_tag else "Unknown Date"

                    reviews.append({
                        "Rating": rating,
                        "Title": title,
                        "Review": review_text,
                        "Reviewer": reviewer_name,
                        "Location": date_location,
                        "Date": date,
                        "Platform": "Flipkart"
                    })

                    if len(reviews) >= num_reviews:
                        break

                except Exception as e:
                    print(f"Error extracting review: {e}")

            page += 1
            time.sleep(1)  # Be polite with delays between requests

        except Exception as e:
            print(f"Error scraping page: {e}")
            break

    return reviews

# Enhanced Amazon scraper
def scrape_amazon_reviews(url, num_reviews=50):
    reviews = []
    page = 1

    while len(reviews) < num_reviews:
        page_url = f"{url}&pageNumber={page}"
        print(f"Scraping: {page_url}")
        
        try:
            html_content = get_page_content(page_url, 'amazon')
            if not html_content:
                break

            soup = BeautifulSoup(html_content, "html.parser")
            review_blocks = soup.find_all("div", {"data-hook": "review"})

            if not review_blocks:
                print("No reviews found on this page.")
                break

            for review_block in review_blocks:
                try:
                    # Extract rating
                    rating_tag = review_block.find("i", {"data-hook": "review-star-rating"})
                    rating_text = rating_tag.get_text(strip=True) if rating_tag else "0 out of 5 stars"
                    rating_match = re.search(r'(\d+\.?\d*) out of 5 stars', rating_text)
                    rating = int(float(rating_match.group(1))) if rating_match else 0

                    # Extract other elements
                    title_tag = review_block.find("a", {"data-hook": "review-title"})
                    title = title_tag.get_text(strip=True) if title_tag else "No Title"

                    review_text_tag = review_block.find("span", {"data-hook": "review-body"})
                    review_text = review_text_tag.get_text(strip=True) if review_text_tag else "No Review Text"

                    name_tag = review_block.find("span", class_="a-profile-name")
                    reviewer_name = name_tag.get_text(strip=True) if name_tag else "Anonymous"

                    date_tag = review_block.find("span", {"data-hook": "review-date"})
                    date = date_tag.get_text(strip=True) if date_tag else "Unknown Date"

                    # Extract verified purchase status
                    verified_tag = review_block.find("span", {"data-hook": "avp-badge"})
                    verified = bool(verified_tag)

                    reviews.append({
                        "Rating": rating,
                        "Title": title,
                        "Review": review_text,
                        "Reviewer": reviewer_name,
                        "Location": "Amazon",
                        "Date": date,
                        "Verified": verified,
                        "Platform": "Amazon"
                    })

                    if len(reviews) >= num_reviews:
                        break

                except Exception as e:
                    print(f"Error extracting review: {e}")

            page += 1
            time.sleep(1)  # Be polite with delays between requests

        except Exception as e:
            print(f"Error scraping page: {e}")
            break

    return reviews

# Enhanced sentiment analysis with fake review detection
def analyze_reviews(reviews):
    sia = get_sentiment_analyzer()
    sentiment_results = {"Positive": 0, "Negative": 0, "Neutral": 0}
    word_list = []
    review_texts = []
    
    # First pass - basic sentiment analysis
    for review in reviews:
        text = review["Review"]
        sentiment_score = sia.polarity_scores(text)["compound"]
        review["SentimentScore"] = sentiment_score

        if sentiment_score >= 0.05:
            sentiment_results["Positive"] += 1
            review["Sentiment"] = "Positive"
        elif sentiment_score <= -0.05:
            sentiment_results["Negative"] += 1
            review["Sentiment"] = "Negative"
        else:
            sentiment_results["Neutral"] += 1
            review["Sentiment"] = "Neutral"

        word_list.extend(re.findall(r"\b[a-z]{3,}\b", text.lower()))
        review_texts.append(text)
    
    # Fake review detection
    if len(review_texts) > 10:  # Only run if we have enough reviews
        vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        X = vectorizer.fit_transform(review_texts)
        
        clf = IsolationForest(contamination=0.1)
        predictions = clf.fit_predict(X)
        
        for i, review in enumerate(reviews):
            review["IsSuspicious"] = predictions[i] == -1
    
    return sentiment_results, word_list, reviews

# Enhanced word cloud generation
def generate_wordcloud(words):
    stopwords = set(nltk.corpus.stopwords.words('english'))
    custom_stopwords = {'product', 'item', 'good', 'great', 'bad', 'like', 'one', 'use', 'would'}
    stopwords.update(custom_stopwords)
    
    filtered_words = [word for word in words if word not in stopwords]
    
    wordcloud = WordCloud(
        width=1200,
        height=600,
        background_color="white",
        colormap="viridis",
        max_words=200,
        contour_width=3,
        contour_color='steelblue'
    ).generate(" ".join(filtered_words))
    
    wordcloud_path = "static/wordcloud.png"
    wordcloud.to_file(wordcloud_path)
    return wordcloud_path

# Price tracking functionality
def track_price(url):
    try:
        html_content = get_page_content(url, 'generic')
        soup = BeautifulSoup(html_content, "html.parser")
        
        # Amazon price extraction
        price_span = soup.find("span", class_="a-price-whole")
        if price_span:
            price = float(price_span.get_text().replace(',', ''))
            return price
        
        # Flipkart price extraction
        price_div = soup.find("div", class_="_30jeq3 _16Jk6d")
        if price_div:
            price = float(price_div.get_text().replace('₹', '').replace(',', ''))
            return price
            
        return None
    except Exception as e:
        print(f"Error extracting price: {e}")
        return None

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        product_url = request.form.get("product_url")
        platform = request.form.get("platform")
        
        if not product_url or not platform:
            return render_template("index.html", error="Please enter a URL and select a platform.")
        
        try:
            # Store product URL in session for tracking
            session['product_url'] = product_url
            session['platform'] = platform
            
            # Scrape reviews
            if platform == "flipkart":
                reviews = scrape_flipkart_reviews(product_url)
            elif platform == "amazon":
                reviews = scrape_amazon_reviews(product_url)
            else:
                return render_template("index.html", error="Invalid platform selected.")
            
            if not reviews:
                return render_template("index.html", error="No reviews found.")
            
            # Get current price
            current_price = track_price(product_url.split('/ref=')[0])  # Get base product URL
            
            # Analyze reviews
            sentiment_results, word_list, analyzed_reviews = analyze_reviews(reviews)
            wordcloud_path = generate_wordcloud(word_list)
            
            # Calculate fake review percentage
            suspicious_count = sum(1 for r in analyzed_reviews if r.get("IsSuspicious", False))
            fake_percentage = (suspicious_count / len(analyzed_reviews)) * 100 if analyzed_reviews else 0
            
            return render_template(
                "index.html",
                reviews=analyzed_reviews[:20],  # Show first 20 reviews
                sentiment_results=sentiment_results,
                wordcloud_path=wordcloud_path,
                current_price=current_price,
                fake_percentage=round(fake_percentage, 1),
                success="Analysis completed successfully!"
            )
            
        except Exception as e:
            print(f"Error during analysis: {e}")
            return render_template("index.html", error=f"Analysis failed: {str(e)}")
    
    return render_template("index.html")

@app.route("/api/analyze", methods=["POST"])
def api_analyze():
    data = request.json
    product_url = data.get("product_url")
    platform = data.get("platform")
    
    if not product_url or not platform:
        return jsonify({"error": "Missing product_url or platform"}), 400
    
    try:
        if platform == "flipkart":
            reviews = scrape_flipkart_reviews(product_url, 20)  # Limit to 20 for API
        elif platform == "amazon":
            reviews = scrape_amazon_reviews(product_url, 20)
        else:
            return jsonify({"error": "Invalid platform"}), 400
        
        if not reviews:
            return jsonify({"error": "No reviews found"}), 404
        
        sentiment_results, word_list, analyzed_reviews = analyze_reviews(reviews)
        current_price = track_price(product_url.split('/ref=')[0])
        
        return jsonify({
            "reviews": analyzed_reviews,
            "sentiment": sentiment_results,
            "current_price": current_price,
            "review_count": len(analyzed_reviews)
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

import os

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
