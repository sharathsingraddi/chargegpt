# ChargeGPT

A conversational assistant for EV charging in Newcastle. You can ask it things like "when is the busiest time to charge?" or "find me the nearest charger to NE4 6PL" and it gives you answers based on real data instead of guessing.

Live app: https://chargegpt.streamlit.app

This was my MSc Data Science dissertation project at Newcastle University (module CSC8639).

## Why I built it

I noticed something while playing around with ChatGPT and Claude. If you ask them specific questions about a place, they often just make up plausible sounding numbers. Ask a plain language model "what are Newcastle's peak EV charging hours?" and it confidently says "4 to 9pm in the evening." That sounds reasonable. It's also wrong.

The real answer, from a dataset of 29,775 actual charging sessions, is 11am with 2,346 sessions. Nothing like the evening.

That kind of made up answer is called hallucination. It's fine if you're chatting for fun. It's a problem if a city planner is deciding where to put new chargers based on it.

So I built ChargeGPT to fix that. Every answer it gives comes from real data. It doesn't guess.

## How it works

There are three layers behind the chat window:

1. An analytics engine. Twelve Python functions that read the actual datasets and calculate the answer.
2. A retrieval system. A vector database (ChromaDB) that stores facts as embeddings, so the system can find the right piece of evidence for any question.
3. An intent classifier. A small language model call that reads your question, works out what type of question it is (about sessions, drivers, stations, a location, or planning), and sends it to the right tool.

The language model itself never invents numbers. It just phrases the answer in a friendly way, using the exact data that was fetched. That's the whole trick.

## The results

I ran an experiment with 25 test questions across three versions of the system:

- Just the plain language model with no data
- The language model with retrieval added
- The full ChargeGPT with everything switched on

| Metric | LLM only | LLM + RAG | Full ChargeGPT |
|---|---|---|---|
| Simple factual queries | 30% | 95% | 100% |
| Complex queries | 0% | 40% | 100% |
| Overall accuracy | 24% | 84% | 100% |
| Hallucination rate | 65% | 0% | 0% |
| Average response time | 2.03s | 1.30s | 2.77s |

The plain language model got 6 out of 25 right and made up a fake number 13 times. The full system got everything right and never invented a number.

## Something interesting I found in the data

Once all three datasets were connected, one gap in Newcastle's charging setup was really obvious.

- 79% of drivers surveyed said they prefer DC fast chargers
- Only 3.5% of Newcastle's stations actually provide DC fast charging
- That's 7 stations out of 198
- Most of those 7 are in NE1 in the city centre. Outer areas like NE5, NE6, NE7 have almost none
- Average driver satisfaction with charging is 2.77 out of 5
- 66% of drivers said they have trouble finding a station that's free

That's the kind of finding a council could actually use.

## The datasets

| Dataset | Size | What's in it |
|---|---|---|
| Charging sessions | 29,775 rows | Energy used, duration, carbon, when it happened |
| Driver survey | 124 people | Preferences, satisfaction, charging habits |
| Newcastle stations | 198 stations | Location, connector type, power, whether you have to pay |

## What the system can do

- Answer general questions about EV charging in Newcastle
- Find the nearest charger to a postcode or a place name (like "nearest charger to Primark")
- Give planning advice on where new stations should go
- Switch between three modes for the evaluation experiment
- Save your chat history if you sign in with an email
- Show you which dataset any answer came from
- Show a map of all 198 stations for location questions

## Tech stack

Python throughout.

For the AI part:
- Anthropic Claude API for the language model
- sentence-transformers (all-MiniLM-L6-v2) for embeddings
- ChromaDB for storing and searching those embeddings

For data:
- pandas for everything
- matplotlib and seaborn for the charts in the dissertation

For the web app:
- Streamlit
- Custom CSS for the dark theme

For location stuff:
- postcodes.io for turning postcodes into coordinates
- OpenStreetMap Nominatim for turning place names into coordinates
- Haversine formula for the actual distance calculation

For deployment:
- GitHub for the code
- Streamlit Community Cloud for hosting

I didn't use LangChain or LlamaIndex. I wrote the retrieval and routing logic myself because I wanted to actually understand what was going on rather than hide it behind a framework.

## Files in this repo

```
chargegpt/
  app.py                          the Streamlit app
  requirements.txt                Python dependencies
  .streamlit/config.toml          Streamlit settings
  usb_features.csv                cleaned session data
  drivers_cleaned.csv             cleaned survey data
  stations_cleaned.csv            cleaned stations data
  knowledge_base.txt              the 32 facts used by RAG
  saved_chats.json                chat history
  evaluation_results.csv          all 25 test questions and results
  evaluation_final_summary.csv    the summary table
  README.md
```

## Running it yourself

Clone the repo:

```
git clone https://github.com/YOUR-USERNAME/chargegpt.git
cd chargegpt
```

Install the dependencies:

```
pip install -r requirements.txt
```

Create a file called `.env` in the folder and put your Anthropic API key in it:

```
ANTHROPIC_API_KEY=your-key-here
```

Then run:

```
streamlit run app.py
```

It opens at http://localhost:8501.

## What I'd do differently or add if I had more time

- Connect to a live availability API so it knows if a charger is currently in use
- Use real driving distance instead of straight line distance
- Move chat history to a proper database (right now it's a JSON file that resets when the app redeploys)
- Test with a bigger set of questions, maybe 200, with more than one person scoring them
- Add route planning ("where should I stop to charge on the way to Edinburgh?")
- Let the system actually book or reserve, not just recommend

Any of these could be a follow up project.

## Related work I looked at

Three papers were closest to what I was doing:

- RecomBot uses RAG to recommend charging stations to drivers
- ChatEV converts charging demand data into text and gets a language model to predict it
- EELLM predicts station occupancy with an explainability layer on top

None of them do all three things I do: conversational chat, serve both drivers and planners in the same system, and actually measure hallucination properly with an experiment.

There is also another paper called ChargeGPT that came out around the same time. It's a completely different thing (forecasting in Shenzhen with GPT-2, no chat). Same name, different project. I cite it in my dissertation to make that clear.

## Author

Sharath Singaraddi
MSc Data Science, School of Engineering, Newcastle University

Supervisors: Dr. Sanchari Deb and Dr. Xinhuan Shu

## Thanks

Thanks to Dr. Deb and Dr. Shu for putting up with a lot of questions and for helping shape the project into something useful.

## License

This is coursework, so please ask before reusing the code.
