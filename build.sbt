name:="machine-learning-examples-by-scalala"

version:="0.0.1"

scalaVersion:="2.9.1"

organization:="everpeace.org"

libraryDependencies  ++= Seq(
            // other dependencies here
            "org.scalala" %% "scalala" % "1.0.0.RC2-SNAPSHOT"
)

resolvers ++= Seq(
            // other resolvers here
            "Scala Tools Snapshots" at "http://scala-tools.org/repo-snapshots/",
            "ScalaNLP Maven2" at "http://repo.scalanlp.org/repo"
)

//compile options
scalacOptions ++= Seq("-unchecked", "-deprecation")