# MicroKG Frontend

React + Vite client for the MicroKG FastAPI service.

## Run Locally

Start the backend first from the project root:

```bash
python api/main.py
```

Then start the frontend:

```bash
npm install
npm run dev
```

The app runs at `http://localhost:5173` and proxies `/api/*` requests to
`http://localhost:8000`.

## Configuration

Use `VITE_API_URL` when the API is not running on the default local port:

```bash
VITE_API_URL=http://localhost:8000 npm run dev
```

## Scripts

- `npm run dev`: start the Vite development server.
- `npm run build`: create a production build.
- `npm run lint`: run ESLint.
- `npm run preview`: preview the production build locally.
