# Cutting Blocks Backend

3D cut optimization system for packing trapezoidal prism-shaped parts into cuboidal stock blocks with automated cutting instruction generation.

## Project Structure

```
backend/
├── src/                          # Core modules
│   ├── geometry/                 # Data classes for shapes
│   ├── packing/                  # Packing algorithms
│   ├── cutting/                  # Cut plan generation
│   ├── optimization/             # Genetic algorithms
│   ├── utils/                    # Configuration
│   └── visualization/            # 3D exports
├── cutting_backend/              # Django project
├── planner/                      # Django app for optimization API
├── step5_mixed_parts.py          # Main hierarchical packing script
├── pack_manually.py              # Manual packing interface
├── requirements.txt              # Python dependencies
└── manage.py                     # Django management
```

## Prerequisites

- Python 3.13.7
- PostgreSQL 13+ (or SQLite for development)

## Installation

### 1. Clone Repository

```bash
git clone <repository-url>
cd backend
```

### 2. Create Virtual Environment

```bash
python -m venv cutting_block_env
source cutting_block_env/bin/activate  # Linux/Mac
# or
cutting_block_env\Scripts\activate  # Windows
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Environment Configuration

Create `.env` file in the backend directory:

```bash
# Django settings
SECRET_KEY=your-secret-key-here
DEBUG=True
ALLOWED_HOSTS=localhost,127.0.0.1

# Database (SQLite - for development)
DATABASE_URL=sqlite:///db.sqlite3

# Database (PostgreSQL - for production)
# DATABASE_URL=postgresql://user:password@localhost:5432/cutting_blocks_db

# CORS settings
CORS_ALLOWED_ORIGINS=http://localhost:3000,http://127.0.0.1:3000
```

**Generate SECRET_KEY:**
```bash
python -c "from django.core.management.utils import get_random_secret_key; print(get_random_secret_key())"
```

### 5. Database Setup

#### Option A: SQLite (Development)

```bash
python manage.py migrate
python manage.py createsuperuser  # Optional: for admin access
```

#### Option B: PostgreSQL (Production)

```bash
# Install PostgreSQL and create database
sudo -u postgres psql
CREATE DATABASE cutting_blocks_db;
CREATE USER cutting_user WITH PASSWORD 'your_password';
GRANT ALL PRIVILEGES ON DATABASE cutting_blocks_db TO cutting_user;
\q

# Update .env with PostgreSQL DATABASE_URL
# Then run migrations
python manage.py migrate
python manage.py createsuperuser
```

## Running the Application

### Django API Server

```bash
python manage.py runserver
```

API will be available at: `http://localhost:8000`

**API Documentation:** `http://localhost:8000/api/schema/swagger-ui/`

### Command-Line Packing Script

#### Interactive Mode

```bash
python step5_mixed_parts.py -i
```

Follow prompts to:
1. Select stock block size
2. Choose parts (from G1-G56 or upload Excel)
3. Set output folder name

#### Non-Interactive Mode (Default Configuration)

```bash
python step5_mixed_parts.py
```

Uses default settings (all G1-G56 parts, size_2 stock block).

### Manual Packing Interface

```bash
python pack_manually.py
```

Interactive interface for manual part placement and testing.

## Input Files

### Excel Format for Custom Parts

Create Excel file with columns:
- `MARK`: Part identifier (e.g., M1, M2)
- `A(W1)`: Width at base (mm)
- `B(W2)`: Width at top (mm)
- `D(length)`: Depth (mm)
- `Thickness`: Height (mm)
- `(α)`: Taper angle (degrees) - optional, defaults to 2.168°

Example: See `example.xlsx`

## Output Files

### Hierarchical Packing Outputs

Located in `outputs/visualizations/step5_mixed_parts/<folder_name>/`:

- `hierarchical_<config>_step1_subblocks.html` - Sub-blocks visualization
- `hierarchical_<config>_step2_complete.html` - Complete packing
- `hierarchical_packing_summary.txt` - Text report with all configurations

### Visualization Color Scheme

- **Black wireframe**: Stock block boundary
- **Green**: Primary parts (desired shapes)
- **Red**: Scrap/waste blocks
- **Blue/Purple/Orange**: Sub-block parts (different colors per sub-block)

## API Endpoints

### Main Endpoints

**POST** `/api/planner/optimize/`
- Optimize packing for given parts and stock
- Body: `{ "stock_block": "size_1", "parts": [...] }`

**POST** `/api/planner/top-3/`
- Get top 3 optimized configurations
- Body: `{ "parts": [...] }`

**GET** `/api/planner/stock-blocks/`
- List available stock block sizes

### Example API Request

```bash
curl -X POST http://localhost:8000/api/planner/top-3/ \
  -H "Content-Type: application/json" \
  -d '{
    "parts": [
      {"name": "G14", "quantity": 20},
      {"name": "G15", "quantity": 10}
    ]
  }'
```

## Configuration

### Stock Blocks

Defined in `src/utils/config.py`:
- `size_1`: 500×500×2000 mm
- `size_2`: 800×500×2000 mm

### Part Specifications

Default parts: G1-G56 defined in `src/utils/config.py`

Custom parts: Load from Excel via interactive mode

### Optimization Parameters

Edit `src/utils/config.py`:
```python
OPTIMIZATION = {
    'population_size': 100,
    'num_generations': 50,
    'mutation_rate': 0.2
}
```

## Troubleshooting

### CadQuery Installation Issues

```bash
pip install --upgrade cadquery
# or
conda install -c conda-forge cadquery
```

### PostgreSQL Connection Error

Check:
1. PostgreSQL service is running
2. Database credentials in `.env` are correct
3. User has proper permissions

### Import Errors

```bash
# Ensure virtual environment is activated
source cutting_block_env/bin/activate

# Reinstall dependencies
pip install -r requirements.txt
```

### Port Already in Use

```bash
# Use different port
python manage.py runserver 8001
```

## Development

### Run Tests

```bash
python test_guillotine_validator.py
python test_top3_api.py
```

### Access Django Admin

1. Create superuser: `python manage.py createsuperuser`
2. Navigate to: `http://localhost:8000/admin`

### Database Migrations

After model changes:
```bash
python manage.py makemigrations
python manage.py migrate
```

