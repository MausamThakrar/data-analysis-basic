import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def load_data(csv_path: str) -> pd.DataFrame:
    """Load sales data from a CSV file."""
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"CSV file not found at {path.resolve()}")
    df = pd.read_csv(path, parse_dates=["date"])
    return df


def add_derived_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add calculated columns such as total_revenue."""
    df = df.copy()
    df["total_revenue"] = df["units_sold"] * df["unit_price"]
    return df


def basic_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """Return summary statistics for numeric columns."""
    return df[["units_sold", "unit_price", "total_revenue"]].describe()


def revenue_by_region(df: pd.DataFrame) -> pd.Series:
    """Aggregate total revenue by sales region."""
    return df.groupby("region")["total_revenue"].sum().sort_values(ascending=False)


def revenue_by_product(df: pd.DataFrame) -> pd.Series:
    """Aggregate total revenue by product."""
    return df.groupby("product")["total_revenue"].sum().sort_values(ascending=False)


def plot_revenue_by_region(agg: pd.Series) -> None:
    """Create a bar chart for revenue by region."""
    plt.figure()
    agg.plot(kind="bar")
    plt.title("Total Revenue by Region")
    plt.xlabel("Region")
    plt.ylabel("Revenue")
    plt.tight_layout()
    plt.show()


def plot_units_sold_histogram(df: pd.DataFrame) -> None:
    """Plot a histogram of units sold."""
    plt.figure()
    df["units_sold"].plot(kind="hist", bins=5)
    plt.title("Distribution of Units Sold")
    plt.xlabel("Units Sold")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.show()


def main():
    # 1. Load data
    df = load_data("data/sample_sales.csv")

    # 2. Add derived columns
    df = add_derived_columns(df)

    # 3. Show basic statistics
    print("=== BASIC STATISTICS ===")
    print(basic_statistics(df))

    # 4. Aggregate revenue
    print("\n=== REVENUE BY REGION ===")
    print(revenue_by_region(df))

    print("\n=== REVENUE BY PRODUCT ===")
    print(revenue_by_product(df))

    # 5. Visualisations
    plot_revenue_by_region(revenue_by_region(df))
    plot_units_sold_histogram(df)


if __name__ == "__main__":
    main()
