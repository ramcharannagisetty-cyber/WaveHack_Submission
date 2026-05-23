# 🧫 PetriAI — AI-Powered Microbial Analysis Platform

Upload a petri dish photo → get colony counts, growth density, contamination flags, and a lab report in seconds.

---

## Quick Start (3 steps)

### Step 1 — Check you have Python 3.8+
```bash
python3 --version
```

### Step 2 — Install dependencies (one time only)
```bash
pip install flask opencv-python-headless pillow numpy
```
> On Ubuntu/Debian: `pip install --break-system-packages flask opencv-python-headless pillow numpy`

### Step 3 — Start the server
```bash
python3 start.py
```

Open your browser at **http://localhost:5050** — that's it!

---

## Project Structure

```
petriai/
├── start.py              ← Run this to start everything
├── petriai.db            ← SQLite database (auto-created)
├── uploads/              ← Uploaded images stored here
├── backend/
│   └── app.py            ← Flask server + CV analysis engine
└── frontend/
    └── index.html        ← Full single-page app
```

---

## What it does

| Feature | How |
|---|---|
| Colony detection | OpenCV contour analysis + adaptive thresholding |
| Dish boundary | Hough circle transform |
| Contamination flags | 14-feature morphology classifier |
| CFU estimate | Colony count × dilution factor |
| Lab reports | Rule-based AI report generator |
| History | SQLite database, persists across restarts |

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/api/health` | Server health check |
| POST | `/api/analyze` | Upload image → analysis JSON |
| GET | `/api/analyses` | List all past analyses |
| GET | `/api/analyses/:id` | Get single analysis |
| GET | `/api/stats` | Aggregate statistics |

### Example: analyze via curl
```bash
curl -X POST http://localhost:5050/api/analyze \
  -F "file=@my_petri_dish.jpg"
```

---

## Upgrading to real AI (Claude/OpenAI reports)

1. Get an Anthropic API key from https://console.anthropic.com
2. Install the client: `pip install anthropic`
3. In `backend/app.py`, replace `generate_report_local()` call with:

```python
import anthropic

def generate_report_ai(data):
    client = anthropic.Anthropic(api_key="sk-ant-...")
    msg = client.messages.create(
        model="claude-opus-4-5",
        max_tokens=1024,
        messages=[{
            "role": "user",
            "content": f"Write an ISO 7218-style microbiology lab report for: {data}"
        }]
    )
    # parse and return structured report
```

---

## Deploying online (free)

### Railway (easiest)
1. Push this folder to a GitHub repo
2. Go to railway.app → New Project → Deploy from GitHub
3. Set start command: `python3 backend/app.py`
4. Done — Railway gives you a public URL

### Render
1. Go to render.com → New Web Service
2. Connect GitHub repo
3. Build command: `pip install flask opencv-python-headless pillow numpy`
4. Start command: `python3 backend/app.py`

---

## Troubleshooting

**"Address already in use"**
```bash
lsof -i :5050     # find what's using the port
kill -9 <PID>     # kill it
python3 start.py  # restart
```

**Images not analyzing correctly**
- Ensure good lighting when photographing petri dishes
- Image should be >300px across
- Circular dish should fill >60% of the frame

**API shows "offline" in browser**
- Make sure `python3 start.py` is running in a terminal
- Check there are no errors in the terminal output
# WaveHack_Submission
