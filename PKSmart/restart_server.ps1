Write-Host "🛑 Cleaning up old PKSmart servers..." -ForegroundColor Yellow

# Find container ID using port 8000 and kill it
$container = docker ps -q --filter "publish=8000"
if ($container) {
    Write-Host "Found running container: $container. Killing it..." -ForegroundColor Red
    docker rm -f $container
} else {
    Write-Host "No conflicts found on port 8000." -ForegroundColor Green
}

# Also cleanup any stopped containers named pksmart_app just in case
docker rm -f pksmart_app 2> $null

Write-Host "🚀 Starting PKSmart Platform..." -ForegroundColor Cyan
docker run --name pksmart_app --rm -p 8001:8000 -v ${PWD}:/app pksmart uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
