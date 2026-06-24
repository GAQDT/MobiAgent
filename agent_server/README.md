# MobiAgent Agent Server

PC-side controller for `OpenHarmonyLocalAgent`.

The board connects to this server over WebSocket. The server receives board
observations, calls the MobiAgent-compatible model endpoint, sends actions back
to the board, and stores the task trace.

## Run

```powershell
cd D:\codes\CSDI\MobiAgent\agent_server
pip install -r requirements.txt
python app.py --host 0.0.0.0 --port 8088 `
  --decider-api-base http://0.tcp.jp.ngrok.io:23909/v1 `
  --grounder-api-base http://0.tcp.jp.ngrok.io:23909/v1 `
  --decider-model MobiMind-Mixed-4B-1031 `
  --grounder-model MobiMind-Mixed-4B-1031 `
  --output-dir .\results
```

Open:

```text
http://<PC-IP>:8088
```

Set the board app server URL to:

```text
ws://<PC-IP>:8088/ws?device_id=openharmony-board-01
```

## Protocol

Board to server:

- `register`
- `observation`
- `result`
- `log`
- `heartbeat`
- `task_request`

Server to board:

- `hello`
- `observe`
- `action`
- `task_update`
- `task_done`
- `stop`
- `error`
