# Unbuffered so log lines show up in `docker logs` immediately. The 2-minute timeout covers a panorama download plus
# inference; gunicorn's 30-second default would kill the worker and force every model to reload.
export PYTHONUNBUFFERED=1
# One log line per request, rejected ones included: who asked (the address the https proxy passes on, since the proxy
# is the only caller this server sees itself), when, what, the status, the response size and the time taken in ms.
ACCESS_LOG_FORMAT='%({x-forwarded-for}i)s %(t)s "%(r)s" %(s)s %(b)s %(M)sms'
# exec so gunicorn gets docker stop's signal and can finish requests in progress.
exec gunicorn -w 1 -b 0.0.0.0:5000 --timeout 120 --access-logfile - --access-logformat "$ACCESS_LOG_FORMAT" \
     "main:create_app()"
