#!/bin/bash

echo "🚀 Multimedia Search Engine - Complete Deployment Script"
echo "========================================================"
echo "UMONS 2024-2025 | Abdelhadi Agourzam & Mohammed El-Ismayily"
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_status() { echo -e "${BLUE}[INFO]${NC} $1"; }
print_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
print_warning() { echo -e "${YELLOW}[WARNING]${NC} $1"; }
print_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# Step 1: Check Docker installation
print_status "Checking Docker installation..."
if ! command -v docker &> /dev/null; then
    print_error "Docker is not installed. Please install Docker first."
    exit 1
fi

if ! command -v docker-compose &> /dev/null; then
    print_error "Docker Compose is not installed. Please install Docker Compose first."
    exit 1
fi

if ! docker info > /dev/null 2>&1; then
    print_error "Docker is not running. Please start Docker first."
    exit 1
fi

print_success "Docker and Docker Compose are ready"

# Step 2: Check required files
print_status "Checking required files..."
required_files=("app.py" "Dockerfile" "docker-compose.yml" "nginx.conf" "requirements.txt" "features" "image.orig")
missing_files=()

for file in "${required_files[@]}"; do
    if [ ! -e "$file" ]; then
        missing_files+=("$file")
    else
        print_success "Found: $file"
    fi
done

if [ ${#missing_files[@]} -ne 0 ]; then
    print_error "Missing required files/directories: ${missing_files[*]}"
    exit 1
fi

# Check content of directories
feature_count=$(ls features/*.pkl 2>/dev/null | wc -l)
image_count=$(ls image.orig/* 2>/dev/null | wc -l)
print_success "Found $feature_count feature files and $image_count images"

# Step 3: Complete Docker cleanup
print_status "Performing complete Docker cleanup..."

# Stop all containers
docker stop $(docker ps -q) 2>/dev/null || true

# Remove all containers
docker rm $(docker ps -aq) 2>/dev/null || true

# Remove all images
docker rmi $(docker images -q) 2>/dev/null || true

# Remove all volumes
docker volume rm $(docker volume ls -q) 2>/dev/null || true

# Remove custom networks
docker network rm $(docker network ls -q --filter type=custom) 2>/dev/null || true

# Clean build cache
docker builder prune -f > /dev/null 2>&1

# Final cleanup
docker system prune -a -f --volumes > /dev/null 2>&1

print_success "Docker environment cleaned"

# Step 4: Check and free required ports
print_status "Checking required ports (80, 5000, 6379)..."

required_ports=(80 5000 6379)
for port in "${required_ports[@]}"; do
    if lsof -i :$port > /dev/null 2>&1; then
        print_warning "Port $port is in use. Attempting to free it..."
        
        # Kill processes using the port
        pids=$(lsof -ti :$port)
        if [ ! -z "$pids" ]; then
            echo $pids | xargs kill -9 2>/dev/null || true
            sleep 2
            
            # Check again
            if lsof -i :$port > /dev/null 2>&1; then
                print_error "Could not free port $port. Please manually stop the process using this port."
                print_error "Use: sudo lsof -i :$port and sudo kill -9 <PID>"
                exit 1
            else
                print_success "Port $port freed"
            fi
        fi
    else
        print_success "Port $port available"
    fi
done

# Step 5: Deploy the application
print_status "Deploying multimedia search engine..."

# Build and start services
docker-compose up --build -d

# Step 6: Wait for services to be ready
print_status "Waiting for services to start..."

# Wait for Redis
print_status "Waiting for Redis..."
retry_count=0
while ! docker-compose exec redis redis-cli ping 2>/dev/null | grep -q PONG; do
    echo -n "."
    sleep 2
    retry_count=$((retry_count + 1))
    if [ $retry_count -gt 30 ]; then
        print_error "Redis failed to start within 60 seconds"
        docker-compose logs redis
        exit 1
    fi
done
print_success "Redis is ready"

# Wait for Flask app
print_status "Waiting for Flask application..."
retry_count=0
while ! curl -f http://localhost:5000/health > /dev/null 2>&1; do
    echo -n "."
    sleep 3
    retry_count=$((retry_count + 1))
    if [ $retry_count -gt 20 ]; then
        print_error "Flask app failed to start within 60 seconds"
        docker-compose logs multimedia_app
        exit 1
    fi
done
print_success "Flask application is ready"

# Wait for Nginx
print_status "Waiting for Nginx..."
retry_count=0
while ! curl -f http://localhost/health > /dev/null 2>&1; do
    echo -n "."
    sleep 2
    retry_count=$((retry_count + 1))
    if [ $retry_count -gt 15 ]; then
        print_error "Nginx failed to start within 30 seconds"
        docker-compose logs nginx
        exit 1
    fi
done
print_success "Nginx is ready"

# Step 7: Verify deployment
print_status "Verifying deployment..."

# Get health status
health_response=$(curl -s http://localhost/health)
echo ""
print_success "Health check response:"
echo "$health_response" | python3 -m json.tool 2>/dev/null || echo "$health_response"

# Step 8: Show deployment summary
echo ""
echo "🎉 DEPLOYMENT SUCCESSFUL! 🎉"
echo "=========================="
echo ""
print_success "Your Multimedia Search Engine is now running!"
echo ""
echo "🌐 Access URLs:"
echo "  Main Application: http://localhost"
echo "  Direct Flask App: http://localhost:5000"
echo "  Health Check: http://localhost/health"
echo ""
echo "🔐 Login Credentials:"
echo "  admin / password123"
echo "  researcher / umons2024"
echo "  student / multimedia"
echo "  user1 / demo123"
echo ""
echo "📊 Container Status:"
docker-compose ps
echo ""
echo "🛠️ Management Commands:"
echo "  View logs: docker-compose logs -f"
echo "  View specific service logs: docker-compose logs <service_name>"
echo "  Stop application: docker-compose down"
echo "  Restart application: docker-compose restart"
echo ""
echo "🔧 Services:"
echo "  - Redis: Authentication and session storage"
echo "  - Flask: Main application with AI models"
echo "  - Nginx: Reverse proxy and load balancer"
echo ""
echo "📈 Features Available:"
echo "  - 3 AI Models: VGG16, ResNet50, MobileNet"
echo "  - $image_count images loaded"
echo "  - 4 similarity metrics: Euclidean, Cosine, Chi-square, Bhattacharyya"
echo "  - Model combination capabilities"
echo "  - Precision-Recall analysis"
echo ""

# Optional: Open in browser
if command -v xdg-open &> /dev/null; then
    read -p "Open application in browser? (y/n): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        xdg-open http://localhost
    fi
elif command -v open &> /dev/null; then
    read -p "Open application in browser? (y/n): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        open http://localhost
    fi
fi

print_success "Deployment completed successfully!"
echo "Happy searching! 🔍✨"