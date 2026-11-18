#!/bin/bash
# Compile the models.
# usage: ./build.sh            # builds main_seq and main_joint
#        ./build.sh seq        # builds just one (seq or joint)
# Expects Fleet and cppstr cloned next to this folder (../Fleet, ../cppstr);
# override with FLEET=/path/to/Fleet CPPSTR=/path/to/cppstr. Compiler: CXX (default g++-15).
set -e
cd "$(dirname "$0")"

CXX="${CXX:-g++-15}"
F="${FLEET:-../Fleet}"
C="${CPPSTR:-../cppstr}"
[ -d "$F/src" ]       || { echo "Fleet not found at $F (see README)"; exit 1; }
[ -f "$C/cppstr.h" ]  || { echo "cppstr not found at $C (see README)"; exit 1; }

# Homebrew puts Eigen at <prefix>/include/eigen3; Fleet includes <eigen3/Eigen/Core>
BREW_INC=""
command -v brew >/dev/null && BREW_INC="-I$(brew --prefix)/include"

INC=()
for d in "" Inference Hypotheses VirtualMachine VirtualMachine/Program Statistics Grammar Grammar/Enumeration \
         Data Containers Threads Dependencies Inference/MCTS Inference/MCMC Inference/Experimental \
         GrammarInference Hypotheses/StochasticVariables; do
	INC+=("-I$F/src/$d")
done

for m in ${@:-seq joint}; do
	echo "building main_$m ..."
	$CXX "Main_$m.cpp" -o "main_$m" -O2 -pthread -std=c++26 -fcoroutines -fopenmp "${INC[@]}" "-I$C" $BREW_INC -lm
done
echo "done"
