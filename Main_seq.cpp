// include before anything else
#include "EigenLib.h"

#include <string>
#include <unistd.h> // for getpid
#include <fstream>
#include <sstream>
using S = std::string; // just for convenience

///~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
/// Set up some basic variables (which may get overwritten)
///~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

S alphabet = "abcd"; // the alphabet we use (possibly specified on command line)
const float strgamma = 0.01; // penalty on string length
const size_t MAX_LENGTH = 10; // longest strings cons will handle (must match run_patterns.py)
size_t BAGS = 4; // overwritten in main() from --targets
bool overlapping = true; // flag whether or not we're using the overlap discount
const size_t TOP_K = 20; // how many of each finished bag's best hypotheses can supply the discount

///~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
/// Declare a grammar
/// This requires a template to specify what types they are (and what order they are stored in)
///~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 
#include "Grammar.h"
#include "Singleton.h"

class MyGrammar : public Grammar<S,S,  S,char,int>,
				  public Singleton<MyGrammar> {
public:
	MyGrammar() {
		// WE'RE MAKING A NEW GRAMMAR

		add("pair(%s,%s)", +[](S a, S b) -> S { return (a.length()+b.length() > MAX_LENGTH ? a : a+b); });
		add("'a'", +[]() -> S { return S("a"); });
		add("'b'", +[]() -> S { return S("b"); });
		add("'c'", +[]() -> S { return S("c"); });
		add("'d'", +[]() -> S { return S("d"); });
		add("rep(%s,%s)",  +[](int n, S s) -> S {
            if(n <= 0)                        return S("");
            if(s.length()*(size_t)n > MAX_LENGTH) return s;
            S out; out.reserve(s.length()*n);
            for(int i = 0; i < n; i++) out += s;
            return out;
        });
		add("1", +[]() -> int { return 1; }, 1.0/1.0);
        add("2", +[]() -> int { return 2; }, 1.0/2.0);
        add("3", +[]() -> int { return 3; }, 1.0/3.0);
        add("4", +[]() -> int { return 4; }, 1.0/4.0);
        add("5", +[]() -> int { return 5; }, 1.0/5.0);
        add("6", +[]() -> int { return 6; }, 1.0/6.0);
        add("7", +[]() -> int { return 7; }, 1.0/7.0);
        add("8", +[]() -> int { return 8; }, 1.0/8.0);
	}
} grammar; 

///~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
/// Overlap helpers and frozen-bag state
///~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#include "StochasticLOTHypothesis.h"

// copies only where x and y overlap (position-aligned; first mismatch cuts the subtree)
Node copy_overlap(Node& x, Node& y) {
	if(x.rule == y.rule) {
		Node out = grammar.makeNode(x.rule);
		
		for(size_t i=0;i<x.nchildren();i++) {
			out.set_child(i, copy_overlap(x.child(i), y.child(i)));
		}
		
		return out;
	}
	else {
		return grammar.makeNode(NullRule);
	}
}

// State describing the earlier (frozen) bags. Set by main() before each search;
// read-only while MCMC runs.
//   frozen_trees[i] = the top-K program trees (best first) saved from earlier bag i
std::vector<std::vector<Node>> frozen_trees;

///~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
/// Declare our hypothesis type
///~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

// Declare a hypothesis class
class InnerHypothesis final : public StochasticLOTHypothesis<InnerHypothesis,S,S,MyGrammar,&grammar> {
public:
	using Super =  StochasticLOTHypothesis<InnerHypothesis,S,S,MyGrammar,&grammar>;
	using Super::Super; // inherit the constructors
	
	static inline double regenerate_p = 0.75;
		
	double compute_single_likelihood(const datum_t& x) override {	
		const auto log_A = log(alphabet.size());
		
		// This would be a normal call:
		auto longout = call(x.input, ""); 
		
		// So now, longout contains much longer strings -- we want to sum up only to the length of x.output
		DiscreteDistribution<S> out;
		for(const auto& [s,lp] : longout) {
			out.addmass(s.substr(0,x.output.length()), lp); 
		}
		const auto outZ = out.Z();
		
		// Likelihood comes from all of the ways that we can delete from the end and the append to make the observed output. 
		double lp = -infinity;
		for(auto& [s,slp] : out.values()) { // add up the probability from all of the strings
			lp = logplusexp(lp, (slp-outZ) + p_delete_append<strgamma,strgamma>(s, x.output, log_A));
		}
		
		return lp;
	}
	
	virtual double compute_prior() override {
		double p = Super::compute_prior(); // plain grammar prior of this program
		
		if(overlapping and not frozen_trees.empty()) {
			const size_t n = frozen_trees.size();
			Node cov = value;                         // c ∩ reps chosen so far
			Node rov = grammar.makeNode(NullRule);    // reps chosen so far, intersected (without c)
			
			for(size_t i=0;i<n;i++) {
				auto& cands = frozen_trees[i];
				
				// find the saved hypothesis of bag i that shares the most with cov
				size_t best_k = 0;
				double best_o = infinity;
				for(size_t k=0;k<cands.size();k++) {
					Node t = copy_overlap(cov, cands[k]);
					double o = grammar.log_probability(t);
					if(o < best_o) { best_o = o; best_k = k; }
				}
				
				cov = copy_overlap(cov, cands[best_k]);
				if(i == 0) rov = cands[best_k];
				else       rov = copy_overlap(rov, cands[best_k]);
			}
			
			double o_c = grammar.log_probability(cov);
			assert(o_c <= 0.0);
			assert(valid(o_c)); // not NaN or inf or -inf
			p -= (double)n * o_c;
			
			if(n >= 2) {
				double o_r = grammar.log_probability(rov);
				assert(o_r <= 0.0);
				assert(valid(o_r));
				p += ((double)n - 1.0) * o_r;
			}
		}
		
		this->prior = p;
		return p;
	}
	
	/**
	 * @brief We'll define a custom proposer here with a global static variable that says how much we 
	 *        regenerate vs insert/delete (defaultly regenerate_p=0.5)
	 * @return 
	 */
	
	[[nodiscard]] virtual std::optional<std::pair<InnerHypothesis,double>> propose() const override {
		std::pair<Node,double> x;
		if(flip(regenerate_p)) {
			auto p = Proposals::regenerate(&grammar, value);	
			if(not p) return {};
			x = p.value();
		}
		else {
			auto p = flip() ? Proposals::insert_tree(&grammar, value) :
							  Proposals::delete_tree(&grammar, value);	
			if(not p) return {};
			x = p.value();
		}
		return std::make_pair(InnerHypothesis(std::move(x.first)), x.second); 		
	}	
//	
	void show(std::string prefix="") override {
		// we're going to make this print by showing the language we created on the line before
		prefix = prefix+"#\n#" +  this->call("", "<err>").string() + "\n";
		Super::show(prefix); 
	}
};

////////////////////////////////////////////////////////////////////////////////////////////

// if we define DO_NOT_INCLUDE_MAIN then we can import everything *except* the below
#ifndef DO_NOT_INCLUDE_MAIN

#include "TopN.h"
#include "MCMCChain.h"
#include "ChainPool.h"
#include "ParallelTempering.h"
#include "Fleet.h" 
#include "Builtins.h"
#include "VMSRuntimeError.h"

#include "HillClimbing.h"

int main(int argc, char** argv){ 
	
	FleetArgs::top_print_best = true; // default print best
	
	// command-line inputs for run_patterns.py
	S targets_arg  = "aaaa,bbbb,cccc,dddd"; // comma-separated bag targets
	S results_path = "results2.tsv";        // where to write the output TSV
	
	// default include to process a bunch of global variables: mcts_steps, mcc_steps, etc
	Fleet fleet("A simple, one-factor formal language learner");
	fleet.add_option("-a,--alphabet", alphabet, "Alphabet we will use"); 	// add my own args
	fleet.add_option("--targets", targets_arg, "Comma-separated bag targets, e.g. aaaa,bbbb");
	fleet.add_option("--results", results_path, "Output TSV path");
	fleet.initialize(argc, argv);
	
	// since we are only storing the top, we can ignore repeats from MCMC 
	FleetArgs::MCMCYieldOnlyChanges = true;
	
	//------------------
	// set these global variables to make VMS a little faster, less accurate
	//------------------
	
	VirtualMachineControl::MAX_RECURSE = 10;
	VirtualMachineControl::MAX_STEPS = 64; // 256;
	VirtualMachineControl::MAX_OUTPUTS = 256;
	
	//------------------
	// set up the data
	//------------------
	
	// Parse --targets into one string per bag. The grammar only has 'a'-'d' and can't
	// build strings longer than MAX_LENGTH, so reject anything else up front.
	std::vector<S> targets;
	{
		std::stringstream ss(targets_arg);
		S t;
		while(std::getline(ss, t, ',')) {
			assert(t.length() <= MAX_LENGTH);                       // grammar can't build longer
			for(char c : t) assert(S("abcd").find(c) != S::npos);   // grammar only has a-d
			targets.push_back(t);
		}
	}
	BAGS = targets.size();
	assert(BAGS >= 2);

	using Data = std::vector<InnerHypothesis::datum_t>;
	
	auto make_datum = [](const S& o) {
		InnerHypothesis::datum_t x; x.input = ""; x.output = o;
		return x;
	};

	// ------------------
	// Helpers
	// ------------------
	
	// saved[b] = the sorted (best first) top-N hypotheses from bag b's final trial.
	// The best TOP_K of each are what the discount can draw on (see set_frozen).
	std::vector<std::vector<InnerHypothesis>> saved(BAGS);

	// Point the prior's discount at bags 0..n-1 (the top TOP_K saved programs of each).
	auto set_frozen = [&](size_t n) {
		frozen_trees.clear();
		for(size_t i=0;i<n;i++) {
			std::vector<Node> trees;
			for(size_t k=0;k<saved[i].size() and k<TOP_K;k++) {
				trees.push_back(saved[i][k].get_value());
			}
			frozen_trees.push_back(trees);
		}
	};

	// One MCMC search over ONE program, from a cold random start.
	auto search = [&](TopN<InnerHypothesis>& top, Data& data) {
		InnerHypothesis h0 = InnerHypothesis::sample();
		ParallelTempering samp(h0, data, FleetArgs::nchains, 1.20);
		for(auto& h : samp.run(Control()) | top | thin(FleetArgs::thin)) {
			UNUSED(h);
		}
	};

	// Sorted (best first) copy of a TopN
	auto sorted_hyps = [](TopN<InnerHypothesis>& top) {
		std::vector<InnerHypothesis> v;
		for(auto& h : top.sorted(false)) v.push_back(h);
		return v;
	};

	// ------------------
	// Output
	// ------------------
	std::ofstream out(results_path);
	out << "data\trank\tposterior\tprior\tlikelihood\thypothesis\tprediction\n";

	// Write one row per hypothesis. The prediction is the character at predict_pos of the
	// hypothesis' most probable output string. Hypotheses whose argmax is too short are
	// dropped (never written); rank counts only rows kept.
	auto emit = [&](std::vector<InnerHypothesis>& hyps, const S& desc, size_t predict_pos, bool has_prediction) {
		size_t rank = 0;
		for(auto& h : hyps) {
			S prediction = "NA"; // no prediction: round is finished
			if(has_prediction) {
				S best = h.call(S(""), S("")).argmax();
				if(predict_pos >= best.length()) continue;
				prediction = S(1, best.at(predict_pos));
			}

			S hs = h.string();
			for(auto& ch : hs) if(ch=='\t'||ch=='\n') ch=' ';

			out << desc << "\t" << rank << "\t"
			    << h.posterior << "\t" << h.prior << "\t" << h.likelihood << "\t"
			    << hs << "\t" << prediction << "\n";
			rank++;
		}
	};

	// ------------------
	// Run: one bag at a time. Earlier bags are frozen; they only affect the prior
	// of the bag being searched (via the overlap discount).
	// ------------------
	for(size_t b=0;b<BAGS;b++) {
		
		set_frozen(b); // discount against bags 0..b-1
		
		for(size_t len=1; len<=targets[b].length(); len++) {
			
			// data is ONLY the current bag, truncated to len
			Data mydata;
			mydata.push_back(make_datum(targets[b].substr(0,len)));
			
			TopN<InnerHypothesis> top(1000);
			search(top, mydata);
			auto hyps = sorted_hyps(top);
			
			// `data` column keeps the old format: earlier bags in full, current bag partial
			S desc;
			for(size_t k=0;k<b;k++) desc += str(k) + ":" + targets[k] + " ";
			desc += str(b) + ":" + targets[b].substr(0,len) + " ";
			
			const bool bag_done = (len == targets[b].length());
			
			if(not bag_done) {
				// predict the next character within this bag
				emit(hyps, desc, len, true);
			}
			else {
				saved[b] = hyps; // freeze this bag's top-N
				
				if(b+1 < BAGS) {
					// Predict the first character of bag b+1: search bag b+1 under the
					// prior only (no data), discounted against bags 0..b. The rows written
					// for this trial are from that search.
					set_frozen(b+1);
					Data nodata;
					TopN<InnerHypothesis> next(1000);
					search(next, nodata);
					auto next_hyps = sorted_hyps(next);
					emit(next_hyps, desc, 0, true);
					set_frozen(b); // restore (the next loop iteration sets it again anyway)
				}
				else {
					emit(hyps, desc, 0, false); // all bags finished: prediction NA
				}
			}
		}
	}

	out.close();
}

#endif