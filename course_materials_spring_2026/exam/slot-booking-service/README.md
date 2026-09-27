# VisualGenAI — Exam Booking

Tiny self-hosted web app for students to claim a timeslot for the
**May 18, 2026 · 18:00 – 22:00** exam.

- 20-min slot = defend current grade
- 40-min slot = push for **+1**
- **3 parallel interviewer queues** — each queue runs its own schedule, admin sets the interviewer name + Zoom link per queue
- 1 pick per student, first-come-first-locked
- Zero npm dependencies (uses the built-in Node `http` module)
- Bookings + queue config persisted to `bookings.json` (atomic write)

## Run locally

```bash
cd /home/dbaranchuk/exam-booking
node server.js
# open http://localhost:3000
```

Override defaults with env vars:

```bash
PORT=8080                       # default 3000
ADMIN_TOKEN="long-random-string" # default "visualgenai-admin" — CHANGE BEFORE HOSTING
BOOKINGS_FILE=/data/bookings.json # point at a persistent volume on cloud hosts
```

## Admin: view and delete bookings

Open in a browser:

```text
http://<your-host>/admin.html?token=<ADMIN_TOKEN>
```

You'll see:

- **Queue configuration** — three cards with interviewer name + Zoom link inputs; click **Save queues** to push them to students. Initially blank; fill them in once you know who's on which queue.
- Stats — students booked, defends, upgrades, capacity used (out of 36 = 3 queues × 12 cells)
- **By time** — every booking sorted by slot, shows which queue, each row has a **Delete** button
- **By student** — grouped, shows queue + slot per student

Delete is also exposed as a REST call (useful for scripts):

```bash
# list all bookings (includes hidden id + timestamps)
curl -H "X-Admin-Token: $TOKEN" https://your-host/api/admin/bookings

# delete one
curl -X DELETE -H "X-Admin-Token: $TOKEN" https://your-host/api/bookings/<id>
```

To reset everything: stop the server, `rm bookings.json`, start again.

---

## Hosting

Pick whichever fits — all of these will be up in under 10 minutes.

### Option A — Cloudflare Tunnel (zero deploy, run from your machine)

Fastest if your laptop/workstation can stay online during the booking window.
No signup, no cloud account, no firewall changes.

```bash
# in one terminal
node server.js

# in another, install cloudflared (https://github.com/cloudflare/cloudflared)
# macOS:  brew install cloudflared
# Linux:  download the binary from the releases page

cloudflared tunnel --url http://localhost:3000
# -> prints a https://*.trycloudflare.com URL — share it with students
```

Pros: instant, free, your file `bookings.json` is the source of truth.
Cons: dies when your machine sleeps. Fine for a few-hour booking window.

`ngrok http 3000` works the same way (free tier requires signup, gives a random URL).

### Option B — Render.com (push-to-deploy, easiest cloud option)

1. Push this folder to a new GitHub repo (the `.gitignore` keeps `bookings.json`
   and `node_modules` out).
2. Go to <https://render.com> → **New +** → **Web Service** → connect your repo.
3. Settings:
   - **Runtime**: Node
   - **Build Command**: *(leave blank — there's nothing to install)*
   - **Start Command**: `node server.js`
   - **Environment Variables**:
     - `ADMIN_TOKEN` = a long random string (use `openssl rand -hex 16`)
4. Click **Create Web Service**. You get `https://your-app.onrender.com` in a minute or two.

**Persistence note**: Render's free tier has an ephemeral disk — `bookings.json`
disappears on every deploy or restart. For a one-day exam you can usually get
away with this, but the safe option is the $1/mo "Starter" plan + add a disk
mounted at `/data`, then set env `BOOKINGS_FILE=/data/bookings.json`.

### Option C — Fly.io (free tier, persistent disk)

Best balance of free + reliable. Bookings survive restarts.

```bash
# install: https://fly.io/docs/hands-on/install-flyctl/
fly launch                                       # detects Node, asks a few questions
fly secrets set ADMIN_TOKEN=$(openssl rand -hex 16)
fly volumes create data --size 1 --region <yr>   # 1 GB is plenty
```

Edit `fly.toml` (created by `fly launch`) and add the mount + env:

```toml
[env]
  BOOKINGS_FILE = "/data/bookings.json"

[mounts]
  source = "data"
  destination = "/data"
```

Then:

```bash
fly deploy
fly status               # gives you the https://<app>.fly.dev URL
fly secrets list         # confirm ADMIN_TOKEN is set
```

### Option D — Railway / Glitch / your own VPS

- **Railway** (<https://railway.app>): connect repo, set `ADMIN_TOKEN` env,
  add a volume mounted at `/data` and `BOOKINGS_FILE=/data/bookings.json`.
- **Glitch** (<https://glitch.com>): "Import from GitHub", paste the URL. The
  app runs instantly. Storage is persistent for the project. Note that free
  projects sleep after 5 min — first student to visit wakes it up (~10 s cold start).
- **VPS** (Hetzner ~€4/mo, DigitalOcean $4): `scp` the folder, install Node 20,
  run `node server.js` under `systemd` or `pm2`, put Caddy in front for HTTPS.

### Containers (Render, Fly, Railway, k8s, anywhere)

A minimal `Dockerfile` is included. Build & run locally:

```bash
docker build -t exam-booking .
docker run -p 3000:3000 -v $(pwd)/data:/data \
  -e ADMIN_TOKEN="long-random-string" \
  exam-booking
```

---

## Checklist before sharing the URL with students

- [ ] `ADMIN_TOKEN` is set to something **not** `visualgenai-admin`
- [ ] `https://...` URL works on phone + desktop
- [ ] Open `/admin.html?token=...` and fill in interviewer + Zoom link for each of the 3 queues, then **Save queues**
- [ ] Test a defend booking, a 40-min upgrade booking, and a delete from the admin page
- [ ] Note where bookings live (`bookings.json` on the host) so you can grab them after

## Files

- `server.js` — `/api/state`, `/api/book`, `/api/admin/bookings`, `DELETE /api/bookings/:id`
- `public/index.html`, `public/styles.css`, `public/app.js` — student-facing booking page
- `public/admin.html` — organizer dashboard (token-gated)
- `Dockerfile`, `.gitignore` — for cloud / container deploys
