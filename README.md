# Welcome to your Lovable project

TODO: Document your project here

## Google Calendar Integration

The calendar tab can merge your local booking slots with Google Calendar events.

### 1) Install backend dependencies

Use the project venv Python:

```powershell
& "c:/Users/Om Jagdish Salyan/Downloads/PRAXIS/bookingAgent/bookingAgent/.venv/Scripts/python.exe" -m pip install -r requirements-booking-agent.txt
```

### 2) Create Google OAuth credentials

1. In Google Cloud Console, create an OAuth client (Desktop app or Web app).
2. Enable Google Calendar API.
3. Download the client secret JSON.

Place the file in project root as `google_client_secret.json` or set:

```powershell
$env:GOOGLE_CALENDAR_CLIENT_SECRET_FILE="C:\path\to\client_secret.json"
```

### 3) Set optional env vars (PowerShell)

```powershell
$env:GOOGLE_CALENDAR_REDIRECT_URI="http://127.0.0.1:8000/api/google-calendar/callback"
$env:GOOGLE_CALENDAR_ID="primary"
$env:BOOKING_TIMEZONE="Asia/Kolkata"
```

### 4) Start backend and frontend

```powershell
python -m uvicorn api_server:app --host 127.0.0.1 --port 8000 --reload
npm run dev
```

### 5) Connect from the Calendar tab

Open the calendar popover and click **Connect Google**. A browser window opens for OAuth consent.
After consent, the callback stores token data in `google_token.json` and merged events appear in calendar views.
