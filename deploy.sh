#!/bin/bash
# Production Deployment Script for AI Attendance System
# Run this script to deploy the application with Docker Compose

set -e  # Exit on error

echo "=========================================="
echo "AI Attendance System - Production Deploy"
echo "=========================================="
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}✓${NC} $1"
}

print_error() {
    echo -e "${RED}✗${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}⚠${NC} $1"
}

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    print_error "Docker is not installed. Please install Docker first."
    exit 1
fi
print_status "Docker found"

# Check if Docker Compose is installed
if ! command -v docker-compose &> /dev/null; then
    print_error "Docker Compose is not installed. Please install Docker Compose first."
    exit 1
fi
print_status "Docker Compose found"

# Check if required files exist
echo ""
echo "Checking required files..."
required_files=(
    "main_optimized.py"
    "config.py"
    "camera_worker_optimized.py"
    "face_recognition_engine_onnx.py"
    "database.py"
    "vector_db.py"
    "liveness_detection.py"
    "logging_config.py"
    "requirements.txt"
    "Dockerfile"
    "docker-compose.yml"
)

missing_files=()
for file in "${required_files[@]}"; do
    if [ -f "$file" ]; then
        print_status "$file"
    else
        print_error "$file (missing)"
        missing_files+=("$file")
    fi
done

if [ ${#missing_files[@]} -gt 0 ]; then
    print_error "Missing required files. Cannot proceed with deployment."
    exit 1
fi

# Create required directories
echo ""
echo "Creating directories..."
mkdir -p logs
mkdir -p vector_db
mkdir -p "Employees Images"
print_status "Directories created"

# CRITICAL: Check if database and embeddings exist
echo ""
echo "Checking for database and embeddings..."
if [ ! -f "attendance.db" ]; then
    print_error "attendance.db NOT FOUND!"
    echo ""
    echo "⚠️  CRITICAL: Database must be created before Docker deployment"
    echo ""
    echo "Run this command first:"
    echo "  python enroll_from_api.py --full"
    echo ""
    echo "This will:"
    echo "  • Auto-create attendance.db"
    echo "  • Auto-create vector_db/"
    echo "  • Download employee photos"
    echo "  • Generate face embeddings"
    echo ""
    exit 1
fi
print_status "attendance.db found"

# Check if vector database exists
if [ ! -d "vector_db" ] || [ ! -f "vector_db/embeddings.db" ]; then
    print_error "vector_db/embeddings.db NOT FOUND!"
    echo ""
    echo "⚠️  CRITICAL: Embeddings must be created before Docker deployment"
    echo ""
    echo "Run this command first:"
    echo "  python enroll_from_api.py --full"
    echo ""
    exit 1
fi
print_status "vector_db/embeddings.db found"

# Verify employees are enrolled
if [ -f "check_embeddings.py" ]; then
    employee_count=$(python check_embeddings.py 2>&1 | grep -oP '\d+(?= employees enrolled)' || echo "0")
    if [ "$employee_count" -gt 0 ]; then
        print_status "$employee_count employees enrolled"
    else
        print_warning "No employees enrolled. System may not recognize anyone."
    fi
fi

# Ask user if they want to rebuild
echo ""
read -p "Do you want to rebuild the Docker image? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo ""
    echo "Building Docker image..."
    docker-compose build
    print_status "Docker image built successfully"
fi

# Stop existing container if running
echo ""
echo "Stopping existing containers..."
docker-compose down 2>/dev/null || true
print_status "Old containers stopped"

# Start the service
echo ""
echo "Starting service..."
docker-compose up -d

# Wait for service to be healthy
echo ""
echo "Waiting for service to start..."
max_attempts=30
attempt=0

while [ $attempt -lt $max_attempts ]; do
    if curl -s http://localhost:8000/health > /dev/null 2>&1; then
        print_status "Service is healthy!"
        break
    fi
    attempt=$((attempt + 1))
    echo -n "."
    sleep 2
done

echo ""

if [ $attempt -eq $max_attempts ]; then
    print_error "Service failed to start within timeout"
    echo ""
    echo "Checking logs:"
    docker-compose logs --tail=20
    exit 1
fi

# Display status
echo ""
echo "=========================================="
echo "Deployment Complete!"
echo "=========================================="
echo ""
print_status "Service is running"
echo ""
echo "📊 Quick Links:"
echo "   • API Documentation: http://localhost:8000/docs"
echo "   • System Status:     http://localhost:8000/api/status"
echo "   • Health Check:      http://localhost:8000/health"
echo ""
echo "📹 Camera Streams:"
echo "   • Punch-In:  http://localhost:8000/api/stream/Punch-In%20Camera"
echo "   • Punch-Out: http://localhost:8000/api/stream/Punch-Out%20Camera"
echo ""
echo "🔧 Useful Commands:"
echo "   • View logs:        docker-compose logs -f"
echo "   • Stop service:     docker-compose stop"
echo "   • Restart service:  docker-compose restart"
echo "   • Check status:     docker-compose ps"
echo "   • Shell access:     docker-compose exec fastapi-attendance /bin/bash"
echo ""

# Show current status
echo "Current Status:"
docker-compose ps
echo ""

# Show recent logs
echo "Recent Logs:"
docker-compose logs --tail=10
echo ""

print_status "Deployment successful!"
