# ClearDrive

ClearDrive is a modular license plate recognition pipeline. It captures webcam frames, detects plates with YOLOv8, reads the text with Tesseract OCR, checks the plate against an InfluxDB whitelist, and publishes an AWS SNS event when a whitelisted plate is detected.

## What it does

1. **Capture** — reads frames from your webcam
2. **Detect** — finds license plates in each frame using YOLOv8
3. **Recognize** — extracts and validates plate text with Tesseract OCR
4. **Whitelist** — checks whether the plate exists in InfluxDB Cloud
5. **Event** — publishes a JSON message to an AWS SNS topic (whitelisted plates only)

The demo app (`main.py`) shows two OpenCV preview windows and prints plate results to the console. Press **q** in either window to quit.

---

## Pipeline overview

```
WebcamModule
    ↓  full frame
PlateDetectionModule
    ↓  cropped plate + bbox/confidence
PlateOCRModule
    ↓  plate text (validated against format)
WhiteListModule
    ↓  whitelisted flag + normalized plate
EventModule
    ↓  SNS publish (only when whitelisted)
```

Each stage passes an `ImageFrame` to the next. If a stage fails, it returns `None` and downstream stages are skipped for that frame.

---

## Prerequisites

Install these before running ClearDrive:

| Requirement | Purpose |
|-------------|---------|
| **Python 3.11+** | Runtime |
| **Webcam** | Live video input |
| **Tesseract OCR** | Plate text recognition |
| **InfluxDB Cloud account** | Whitelist storage |
| **AWS account** | SNS event publishing |

### Install Tesseract

**Windows**

1. Download the installer from [UB Mannheim Tesseract](https://github.com/UB-Mannheim/tesseract/wiki).
2. Run the installer (default path: `C:\Program Files\Tesseract-OCR\tesseract.exe`).
3. Either add Tesseract to your `PATH`, or set `TESSERACT_CMD` in `.env` (see [Configuration](#configuration)).

**macOS**

```bash
brew install tesseract
```

**Linux (Debian/Ubuntu)**

```bash
sudo apt update && sudo apt install tesseract-ocr
```

Verify the install:

```bash
tesseract --version
```

---

## Step-by-step setup

### 1. Clone or download the project

```bash
cd C:\Users\ismai\Desktop\ClearDrive
```

### 2. Create a virtual environment

**Windows (PowerShell)**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**macOS / Linux**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Python dependencies

```bash
pip install -r requirements.txt
```

On first run, the plate detection module automatically downloads the YOLOv8 weights file to `models/yolov8_license_plate.pt` (~6 MB).

### 4. Create your environment file

Copy the example file and edit it with your real values:

**Windows**

```powershell
Copy-Item .env.example .env
```

**macOS / Linux**

```bash
cp .env.example .env
```

Open `.env` in a text editor. The sections below explain every variable.

### 5. Set up InfluxDB Cloud (whitelist)

ClearDrive stores whitelisted plates in InfluxDB Cloud.

1. Sign up at [InfluxDB Cloud](https://www.influxdata.com/products/influxdb-cloud/).
2. Create a bucket named `Whitelist` (or use another name and set `INFLUX_BUCKET` in `.env`).
3. Generate an API token with read/write access to that bucket.
4. Copy your cluster URL (e.g. `https://eu-central-1-1.aws.cloud2.influxdata.com`).
5. Add these to `.env`:

```env
INFLUX_URL=https://your-region.aws.cloud2.influxdata.com
INFLUX_TOKEN=your-influx-cloud-api-token
INFLUX_BUCKET=Whitelist
```

`INFLUX_ORG` is optional — ClearDrive auto-discovers it from your token when there is only one organization.

### 6. Seed the whitelist

Edit `scripts/seed_whitelist.py` and add the plates you want to allow:

```python
PLATES = ["SK 8546 AR", "KU 1234 AB"]
```

Run the seed script:

```bash
python scripts/seed_whitelist.py
```

You should see output like:

```
Wrote SK 8546 AR to bucket 'Whitelist'
Plates in bucket: ['SK 8546 AR']
```

### 7. Set up AWS SNS (events)

ClearDrive publishes an SNS message only when a **whitelisted** plate is detected.

#### 7a. Create an SNS topic

1. Open the [AWS SNS console](https://console.aws.amazon.com/sns/).
2. Click **Create topic**.
3. Choose **Standard**, give it a name (e.g. `cleardrive-events`), and create it.
4. Copy the **Topic ARN** (e.g. `arn:aws:sns:us-east-1:123456789012:cleardrive-events`).

#### 7b. Create IAM credentials

Create an IAM user or role with permission to publish to your topic. Minimum policy:

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": "sns:Publish",
      "Resource": "arn:aws:sns:YOUR-REGION:YOUR-ACCOUNT-ID:cleardrive-events"
    }
  ]
}
```

Generate an access key for the IAM user, or use `aws configure` if you prefer the AWS CLI credential file.

#### 7c. Add AWS settings to `.env`

```env
SNS_TOPIC_ARN=arn:aws:sns:us-east-1:123456789012:cleardrive-events
AWS_REGION=us-east-1
AWS_ACCESS_KEY_ID=your-access-key-id
AWS_SECRET_ACCESS_KEY=your-secret-access-key
```

Alternatively, skip the key variables and run:

```bash
aws configure
```

boto3 will read credentials from `~/.aws/credentials`.

### 8. Run the application

```bash
python main.py
```

**What to expect:**

- Two preview windows open: `ClearDrive - Webcam` (full frame) and `ClearDrive - Plate` (cropped plate).
- When a plate is read, the console prints its text and whitelist status.
- Whitelisted plates trigger an SNS publish; the message ID is printed.
- Green bounding box = whitelisted. Red = not whitelisted.
- Press **q** in either window to exit.

**Example console output:**

```
Opening webcam and loading plate detector...
Ready. Press 'q' in the preview window to quit.
Plate: SK 8546 AR (whitelisted)
  SNS event published (message id: abc123-def456-...)
```

---

## Configuration

All settings are loaded from `.env` in the project root. See `.env.example` for the full template.

### OCR settings

| Variable | Default | Description |
|----------|---------|-------------|
| `OCR_PLATE_FORMAT` | `FF 0000 XX` | Expected plate layout. `F` = prefix, `0` = digit, `X` = letter |
| `OCR_PLATE_PREFIX_VALUES` | `SK,KU,ST,TE,VE` | Allowed values for `F` segments |
| `OCR_BLACK_V_MAX` | `80` | HSV value threshold for black character isolation |
| `OCR_BLACK_S_MAX` | `100` | HSV saturation threshold for black character isolation |
| `OCR_MIN_PLATE_HEIGHT` | `50` | Minimum crop height before OCR upscaling |
| `OCR_TESSERACT_PSM` | `7` | Tesseract page segmentation mode (single text line) |
| `TESSERACT_CMD` | *(none)* | Path to `tesseract.exe` on Windows if not on `PATH` |

**Plate format tokens:**

- `F` — must match one of the values in `OCR_PLATE_PREFIX_VALUES` (length must match the run of `F`s)
- `0` — digit
- `X` — letter
- Spaces in the pattern are formatting only (e.g. `FF 0000 XX` → `SK 8546 AR`)

### InfluxDB whitelist settings

| Variable | Default | Description |
|----------|---------|-------------|
| `INFLUX_URL` | *(required)* | InfluxDB Cloud cluster URL |
| `INFLUX_TOKEN` | *(required)* | API token with bucket access |
| `INFLUX_ORG` | auto-discovered | Organization name or ID |
| `INFLUX_BUCKET` | `Whitelist` | Bucket containing whitelist data |
| `INFLUX_WHITELIST_MEASUREMENT` | `whitelist` | InfluxDB measurement name |
| `INFLUX_WHITELIST_FIELD` | `plate` | Field name storing plate text |
| `INFLUX_WHITELIST_QUERY` | *(built-in)* | Optional custom Flux query override |
| `WHITELIST_CACHE_TTL_SECONDS` | `300` | How long to cache plates in memory (seconds) |

### AWS SNS settings

| Variable | Default | Description |
|----------|---------|-------------|
| `SNS_TOPIC_ARN` | *(required)* | ARN of the SNS topic to publish to |
| `AWS_REGION` | `us-east-1` | AWS region for the SNS client |
| `AWS_ACCESS_KEY_ID` | *(from env or CLI)* | IAM access key |
| `AWS_SECRET_ACCESS_KEY` | *(from env or CLI)* | IAM secret key |
| `AWS_SESSION_TOKEN` | *(optional)* | Temporary session token (STS) |

### SNS message format

When a whitelisted plate is detected, ClearDrive publishes a JSON message like:

```json
{
  "plate": "SK 8546 AR",
  "timestamp": "2026-06-06T14:30:00.000000+00:00",
  "confidence": 0.87,
  "bbox": [120, 340, 180, 45],
  "text": "SK 8546 AR",
  "input_source": "whitelist"
}
```

The SNS subject line is `ClearDrive: SK 8546 AR`. Each unique plate is published once per session (duplicate frames are deduplicated).

---

## Project structure

```
ClearDrive/
├── main.py                          # Demo entry point
├── requirements.txt                 # Python dependencies
├── .env.example                     # Environment variable template
├── README.md                        # This file
├── models/
│   └── yolov8_license_plate.pt      # Auto-downloaded YOLO weights
├── scripts/
│   └── seed_whitelist.py            # Seed plates into InfluxDB
└── cleardrive/
    ├── core/
    │   ├── module.py                # Abstract base class for all modules
    │   ├── types.py                 # ImageFrame dataclass
    │   └── config.py                # .env loading and helpers
    └── modules/
        ├── camera/webcam.py         # WebcamModule
        ├── detection/plate.py       # PlateDetectionModule (YOLOv8)
        ├── recognition/plate_ocr.py # PlateOCRModule (Tesseract)
        ├── whitelist/
        │   ├── whitelist.py         # WhiteListModule
        │   └── influx_cache.py      # InfluxDB client + cache
        └── event/
            ├── event.py             # EventModule
            └── sns_publisher.py     # SNS client wrapper
```

---

## Modules

Every module extends `Module` and implements `process(frame) → ImageFrame | None`. Modules support context manager usage (`with Module() as m:`) for automatic `setup()` / `teardown()`.

| Module | Input | Output metadata |
|--------|-------|-----------------|
| `WebcamModule` | — | `device_id`, `width`, `height` |
| `PlateDetectionModule` | full frame | `bbox`, `confidence`, `method` |
| `PlateOCRModule` | cropped plate | `text`, `plate_format` |
| `WhiteListModule` | OCR result | `whitelisted`, `plate` |
| `EventModule` | whitelist result | `event_published`, `sns_message_id`, `event_error` |

### Using modules in your own code

```python
from cleardrive.modules import (
    WebcamModule,
    PlateDetectionModule,
    PlateOCRModule,
    WhiteListModule,
    EventModule,
)

with (
    WebcamModule(device_id=0) as camera,
    PlateDetectionModule() as detector,
    PlateOCRModule() as ocr,
    WhiteListModule() as whitelist,
    EventModule() as events,
):
    frame = camera.process()
    plate = detector.process(frame)
    if plate is not None:
        ocr_result = ocr.process(plate)
        if ocr_result is not None:
            whitelist_result = whitelist.process(ocr_result)
            if whitelist_result is not None:
                events.process(whitelist_result)
```

---

## Troubleshooting

### `ModuleNotFoundError: No module named 'influxdb_client'` (or similar)

Activate the virtual environment and install dependencies:

```bash
.\.venv\Scripts\Activate.ps1   # Windows
pip install -r requirements.txt
```

### `tesseract is not installed or it's not in your PATH`

Install Tesseract (see [Prerequisites](#prerequisites)) and add to `.env`:

```env
TESSERACT_CMD=C:\Program Files\Tesseract-OCR\tesseract.exe
```

### `INFLUX_TOKEN is required`

Set `INFLUX_URL` and `INFLUX_TOKEN` in `.env`. Run `python scripts/seed_whitelist.py` to verify connectivity.

### `AWS credentials are required to publish SNS events`

Add `AWS_ACCESS_KEY_ID` and `AWS_SECRET_ACCESS_KEY` to `.env`, or run `aws configure`. Ensure the IAM user has `sns:Publish` on your topic.

### `SNS publish failed: ...`

Check that `SNS_TOPIC_ARN` matches your topic, `AWS_REGION` is correct, and credentials are valid. The app continues running; the error is printed once per plate.

### Webcam does not open on Windows

ClearDrive uses the DirectShow backend (`CAP_DSHOW`) by default on Windows to avoid long startup hangs. Try a different `device_id` if you have multiple cameras:

```python
WebcamModule(device_id=1)
```

### Plate detected but text is wrong

- Improve lighting and angle toward the plate.
- Adjust `OCR_BLACK_V_MAX` / `OCR_BLACK_S_MAX` if plate characters are not black.
- Confirm `OCR_PLATE_FORMAT` and `OCR_PLATE_PREFIX_VALUES` match your country's plate format.

### Plate text correct but not whitelisted

- Ensure the plate in InfluxDB matches the normalized format (e.g. `SK 8546 AR`, not `SK8546AR`).
- Re-run `python scripts/seed_whitelist.py` after editing `PLATES`.
- Wait up to `WHITELIST_CACHE_TTL_SECONDS` (default 5 min) for the in-memory cache to refresh, or restart the app.

### YOLO model download fails

Download manually from [Hugging Face](https://huggingface.co/orionwambert/yolov8-license-plate-detection/resolve/main/best.pt) and save as:

```
models/yolov8_license_plate.pt
```

---

## Development

### Adding a new module

1. Create a class extending `Module` in `cleardrive/modules/<domain>/`.
2. Set a `name` class attribute.
3. Implement `process()`, and optionally `setup()` / `teardown()`.
4. Export from the domain `__init__.py` and `cleardrive/modules/__init__.py`.
5. Wire it into `main.py` or your own pipeline.

### Extending the pipeline

Modules are chained manually — there is no auto-discovery. Add your module after the stage whose output it needs:

```python
result = my_module.process(previous_result) if previous_result is not None else None
```

---

## License

This project is provided as-is for development and demonstration purposes.
