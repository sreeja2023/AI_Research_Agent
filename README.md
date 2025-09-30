Project Galileo: AI Research Agent
..
Summary
--
Project Galileo is an AI system I built to help people find and understand information faster. Today, researchers and professionals waste a lot of time searching the web, checking sources, and writing reports. Galileo does this automatically, it breaks a big question into smaller ones, searches the web, checks the facts, and writes a clear report with links to the original sources.
How It Works

1. Breaking Down Questions (The Planner)

Galileo takes a big question and splits it into smaller, easier parts.

Example: “What are the top risks to India’s electronics supply chain in the next 12 months?” becomes:

Who are India’s main suppliers?

What are India’s new government policies?

What are the risks from Taiwan and China?

Are there global shipping or logistics problems ahead?

2. Searching and Collecting Data (The Scout)

Galileo uses a search API (Serper.dev) to look up answers online.

It gathers text from reliable websites and ignores ads or junk content.


3. Understanding and Checking Information (The Analyst)

The AI reads through the collected text and pulls out important facts.

It always saves the web link for every fact, so the source is clear.

If different sources disagree, it highlights the contradiction

4. Writing the Report (The Writer)

The AI uses GPT-4 to write a clear report in Markdown format.

Every fact includes a source link right after it.

Example:

> India has launched a $10 billion PLI scheme to support local semiconductor manufacturing [https://www.livemint.com/economy/pli-scheme-semiconductors].



Extra Features

Spotting contradictions: If one source says 15% growth and another says 8%, the report points this out.

JSON output: The report can also be saved in structured JSON for use in other apps.

Follow-up questions: Users can ask Galileo for more details without starting over.


Tools Used
--
Python for coding

LangChain for AI workflow

OpenAI GPT-4 for reasoning and writing

Serper.dev API for search

BeautifulSoup for scraping websites


Results
--
Cut research time by 70% compared to doing it manually.

Reports are clear, evidence-based, and trustworthy.

Useful for analysts, researchers, and decision-makers.
Conclusion

Project Galileo shows how AI can act like a smart research assistant. It saves time, avoids bias, and gives clear answers backed by sources. Instead of drowning in endless search results, users get useful knowledge quickly.
