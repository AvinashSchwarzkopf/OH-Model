There are two versions of the model right now, Main_seq runs a hypothesis on each bag one at a time, this is the one I built and it uses the rep grammar (which only has rep, pair, and literals). Main_joint is the original version that goes over all bags at once, and it uses Steve's formal language theory grammar. The grammars are written in the main files so you can check them out.

This is built on [Fleet](https://github.com/piantado/Fleet)! So you'll have to download it to run this model. If you have a mac, install GCC 15 and Eigen, since Fleet won't run on the built-in compiler. And download cppstr, which is a C++ library Fleet and the models use. Put both in folders named `Fleet` and `cppstr` next to this repo's folder.

To build and run the model:
```bash
./build.sh
./main_seq --targets=aaaa,bbbb,cccc,dddd --results=out.tsv --chains=5 --threads=4 --time=10s
```
You can change chains, threads, and time as needed.