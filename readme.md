# MarginCall_2X - The Bankruptcy Booster

This project is a highly personalized, AI-powered Telegram trading companion bot. Nicknamed "The Bankruptcy Booster," it's designed to assist users in monitoring financial markets, analyzing their portfolio, and managing risk, all delivered in the tone of a trading floor buddy with a dark sense of humor.

## Core Features

*   **AI Trading Companion**:
    *   Powered by the Google Gemini AI, its distinct and strong personality is shaped by the `system_prompt` within `config.py`.
    *   Features a built-in "trash talk" system with humorous and cynical responses that change based on market conditions, making interactions more engaging.

*   **Modular Analysis Engines**:
    *   `engine_market.py`: Handles market data analysis, fetching real-time quotes and news.
    *   `engine_portfolio.py`: Manages and calculates the real-time Profit and Loss (PnL) of the investment portfolio.
    *   `engine_risk.py`: Monitors global macroeconomic indicators and systemic market risks.

*   **Brokerage Integration**:
    *   Integrates with Fubon Securities via the `fubon.py` module to fetch real-time portfolio data from the user's account.

*   **Telegram Interface**:
    *   Built with the `telebot` framework, allowing for easy and convenient interaction on Telegram.
    *   Includes a `/reset` command to clear the AI's conversation history and start fresh.

*   **Strict Safety Protocols**:
    *   Implements a "Double-Check Protocol" that requires secondary user confirmation for all buy, sell, or position modification commands to prevent costly errors.
    *   Enforces a strict "Portfolio Reporting Format" to ensure clear, consistent, and accurate PnL reports every time.

*   **Intelligent Model Management**:
    *   Dynamically switches to the most suitable Gemini AI model based on the current market status (e.g., US or Taiwan market open hours).
    *   Includes an API failover mechanism that automatically switches to a backup model if a service becomes unstable, ensuring high availability.

## Installation & Setup

1.  **Clone the Project**:
    ```bash
    git clone https://github.com/your-username/MarginCall_2X.git
    cd MarginCall_2X
    ```

2.  **Create and Activate a Virtual Environment**:
    ```bash
    python -m venv venv
    source venv/bin/activate  # For Windows: venv\Scripts\activate
    ```

3.  **Install Dependencies**:
    This project does not yet provide a `requirements.txt` file. Please install the necessary packages manually based on the `import` statements in `main.py`:
    ```bash
    pip install python-dotenv pytelegrambotapi google-generativeai
    # ...and any SDKs required for Fubon Securities integration.
    ```

4.  **Set Up Environment Variables**:
    Copy the `.env.un~` file to `.env` and fill in your personal credentials:
    ```bash
    cp .env.un~ .env
    ```
    Then, edit the `.env` file with your information:
    ```env
    TELEGRAM_BOT_TOKEN="Your_Telegram_Bot_Token"
    GEMINI_API_KEY="Your_Gemini_API_Key"
    # ...and any credentials required for the Fubon Securities API.
    ```

## Usage

Once everything is set up, run the main script to start the bot:
```bash
python main.py
```
You can then start chatting with your "Bankruptcy Booster" on Telegram.

---

### **Disclaimer**

This project is for technical demonstration and proof-of-concept purposes only and **does not constitute professional investment advice**. All AI responses and data analysis are for reference only. Users are solely responsible for all risks and outcomes of investment decisions made based on this project. Please conduct thorough research and risk assessment before making any real trades.
