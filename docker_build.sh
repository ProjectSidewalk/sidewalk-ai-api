# Pass "dev" to build a separately tagged image for testing, so the prod image isn't overwritten.
if [ "$1" = "dev" ]; then IMAGE=sidewalk-ai-api-dev; else IMAGE=sidewalk-ai-api; fi
docker  build  -t  $IMAGE  .
