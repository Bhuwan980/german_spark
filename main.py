from fastapi import FastAPI, Request
import httpx

app = FastAPI()

OLLAMA_URL = "http://localhost:11434/api/generate"

@app.post("/chat")
async def chat_with_mistral(req: Request):
    data = await req.json()
    
    user_prompt = data.get("prompt", "")
    system_prompt = (
        "You are a friendly and engaging German language conversation partner. "
        "Focus mainly on keeping the conversation flowing naturally. Correct grammar or usage only when necessary, "
        "and provide short, encouraging feedback. Keep it fun and relaxed, not too formal or overwhelming.\n\n"
        f"User: {user_prompt}\nAI:"
    )

    payload = {
        "model": "mistral",
        "prompt": system_prompt,
        "stream": False
    }

    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(OLLAMA_URL, json=payload)
            response.raise_for_status()
            return response.json()
    except httpx.HTTPStatusError as exc:
        return {"error": f"HTTP error occurred: {exc.response.status_code} - {exc.response.text}", "status_code": exc.response.status_code}
    except httpx.RequestError as exc:
        return {"error": f"An error occurred while requesting Ollama: {str(exc)}", "status_code": 500}
    except Exception as e:
        return {"error": f"An unexpected error occurred: {str(e)}", "status_code": 500}