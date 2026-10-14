# Pass "dev" to build a separately tagged image for testing, so the prod image isn't overwritten.
case "$1" in
  "")  IMAGE=sidewalk-ai-api ;;
  dev) IMAGE=sidewalk-ai-api-dev ;;
  *)   echo "Usage: $0 [dev]" >&2; exit 1 ;;
esac
docker  build  -t  $IMAGE  .
