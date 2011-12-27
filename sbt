java -Dfile.encoding=UTF8 -Xmx1536M -Xss1M -XX:+CMSClassUnloadingEnabled -XX:MaxPermSize=256m -jar `dirname $0`/sbt-launch-0.10.0.jar "$@"
