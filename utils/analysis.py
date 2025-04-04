import os
import pandas as pd
import matplotlib.pyplot as plt


def analyze_data():
    import seaborn as sns

    base_path = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

    # Ensure output directory exists
    output_dir = os.path.join(base_path, "output")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    raw_articles = pd.read_parquet(os.path.join(base_path, "data", "ebnerd", "articles.parquet"))

    # Ensure correct data types
    raw_articles["published_time"] = pd.to_datetime(raw_articles["published_time"], errors="coerce")

    # Function to remove outliers based on IQR
    def remove_outliers(df, column):
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

    # 1) Article Length Distribution (removing outliers)
    plt.figure(figsize=(10, 6))
    raw_articles["article_length"] = raw_articles["body"].str.len()
    raw_articles_no_outliers = remove_outliers(raw_articles, "article_length")
    sns.histplot(raw_articles_no_outliers["article_length"], bins=50, kde=True, color="skyblue")
    plt.xlabel("Article Length (number of characters)", fontsize=20)
    plt.ylabel("Frequency", fontsize=20)
    plt.title("Distribution of Article Lengths (Excluding Outliers)", fontsize=22, fontweight="bold")
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.savefig(os.path.join(output_dir, "article_length_distribution.png"))
    plt.close()

    # 2) Publication Date Distribution
    plt.figure(figsize=(10, 6))
    # Extract year from the publication date and count the number of articles per year
    raw_articles["year"] = raw_articles["published_time"].dt.year
    pub_years = raw_articles["year"].dropna().value_counts().sort_index()
    # Plot the data as a bar plot (binned by year)
    pub_years.plot(kind="bar", color="skyblue")
    plt.xlabel("Publication Year", fontsize=20)
    plt.ylabel("Number of Articles", fontsize=20)
    plt.title("Publication Date Distribution (Binned by Year)", fontsize=22, fontweight="bold")
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()  # Adjust layout to prevent clipping
    plt.savefig(os.path.join(output_dir, "publication_date_distribution_year.png"))
    plt.close()

    # 2) Publication Date Distribution
    plt.figure(figsize=(10, 6))
    # Filter articles from 2023
    articles_2023 = raw_articles[raw_articles["published_time"].dt.year == 2023]
    # Extract the date and count the number of articles per date in 2023
    pub_dates_2023 = articles_2023["published_time"].dropna().dt.date.value_counts().sort_index()
    # Plot the data as a line plot (daily counts)
    pub_dates_2023.plot(kind="line", marker="o", linestyle="-")
    plt.xlabel("Publication Date", fontsize=20)
    plt.ylabel("Number of Articles", fontsize=20)
    plt.title("Publication Date Distribution for 2023", fontsize=22, fontweight="bold")
    plt.xticks(fontsize=18, rotation=45)
    plt.yticks(fontsize=18)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()  # Adjust layout to prevent clipping
    plt.savefig(os.path.join(output_dir, "publication_date_distribution_2023.png"))
    plt.close()

    # 3) Top 20 Categories (bar plot)
    if "category_str" in raw_articles.columns:
        plt.figure(figsize=(12, 6))
        top_categories = raw_articles["category_str"].value_counts().head(20)
        sns.barplot(x=top_categories.values, y=top_categories.index, palette="viridis")
        plt.xlabel("Count", fontsize=20)
        plt.ylabel("Category", fontsize=20)
        plt.title("Top 20 Article Categories", fontsize=22, fontweight="bold")
        plt.xticks(fontsize=18)
        plt.yticks(fontsize=18)
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.tight_layout()  # Adjust layout to prevent clipping
        plt.savefig(os.path.join(output_dir, "top_20_categories.png"))
        plt.close()

    # 4) Top 20 Topics (bar plot)
    if "topics" in raw_articles.columns:
        # Convert topics from comma-separated strings to lists and explode them
        topics_series = (
            raw_articles["topics"]
            .dropna()
            .apply(lambda x: [t.strip() for t in x.split(",")] if isinstance(x, str) else x)
        )
        all_topics = topics_series.explode()
        top_topics = all_topics.value_counts().head(20)
        plt.figure(figsize=(12, 6))
        sns.barplot(x=top_topics.values, y=top_topics.index, palette="magma")
        plt.xlabel("Count", fontsize=20)
        plt.ylabel("Topic", fontsize=20)
        plt.title("Top 20 Topics", fontsize=22, fontweight="bold")
        plt.xticks(fontsize=18)
        plt.yticks(fontsize=18)
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.tight_layout()  # Adjust layout to prevent clipping
        plt.savefig(os.path.join(output_dir, "top_20_topics.png"))
        plt.close()

    # 5) Total_inviews Distribution (removing outliers)
    plt.figure(figsize=(10, 6))
    raw_articles_no_outliers = remove_outliers(raw_articles, "total_inviews")
    sns.histplot(raw_articles_no_outliers["total_inviews"], bins=50, kde=True, color="orange")
    plt.xlabel("Total Inviews", fontsize=20)
    plt.ylabel("Frequency", fontsize=20)
    plt.title("Distribution of Total Inviews (Excluding Outliers)", fontsize=22, fontweight="bold")
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.savefig(os.path.join(output_dir, "total_inviews_distribution.png"))
    plt.close()

    # 6) Total_pageviews Distribution (removing outliers)
    plt.figure(figsize=(10, 6))
    raw_articles_no_outliers = remove_outliers(raw_articles, "total_pageviews")
    sns.histplot(raw_articles_no_outliers["total_pageviews"], bins=50, kde=True, color="purple")
    plt.xlabel("Total Pageviews", fontsize=20)
    plt.ylabel("Frequency", fontsize=20)
    plt.title("Distribution of Total Pageviews (Excluding Outliers)", fontsize=22, fontweight="bold")
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.savefig(os.path.join(output_dir, "total_pageviews_distribution.png"))
    plt.close()

    # 7) Total_read_time Distribution (removing outliers)
    plt.figure(figsize=(10, 6))
    raw_articles_no_outliers = remove_outliers(raw_articles, "total_read_time")
    sns.histplot(raw_articles_no_outliers["total_read_time"], bins=50, kde=True, color="teal")
    plt.xlabel("Total Read Time", fontsize=20)
    plt.ylabel("Frequency", fontsize=20)
    plt.title("Distribution of Total Read Time (Excluding Outliers)", fontsize=22, fontweight="bold")
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.savefig(os.path.join(output_dir, "total_read_time_distribution.png"))
    plt.close()


analyze_data()
