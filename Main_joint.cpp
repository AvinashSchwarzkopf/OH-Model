// include before anything else
#include "EigenLib.h"

#include <string>
#include <unistd.h> // for getpid
using S = std::string; // just for convenience

///~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
/// Set up some basic variables (which may get overwritten)
///~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

S alphabet = "abcd"; // the alphabet we use (possibly specified on command line)
//thread_local S datastr  = "01,01001,010010001,01001000100001"; // the data, comma separated -- this data should be able to get a posterior of -34.9
const float strgamma = 0.01; //75; // penalty on string length
const size_t MAX_LENGTH = 8; // longest strings cons will handle
const size_t BAGS = 4;
bool overlapping = true; //flag whether or not we're using the overlap cost

///~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
/// Declare a grammar
/// This requires a template to specify what types they are (and what order they are stored in)
///~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 
#include "Grammar.h"
#include "Singleton.h"

class MyGrammar : public Grammar<S,S,  S,char,bool>,
				  public Singleton<MyGrammar> {
public:
	MyGrammar() {
		add("tail(%s)",      +[](S s)      -> S { return (s.empty() ? S("") : s.substr(1,S::npos)); });
		add("head(%s)",      +[](S s)      -> S { return (s.empty() ? S("") : S(1,s.at(0))); });
//
//		add_vms<S,S,S>("append(%s,%s)",  new std::function(+[](MyGrammar::VirtualMachineState_t* vms, int) {
//			S b = vms->getpop<S>();
//			S& a = vms->stack<S>().topref();
//			
//			if(a.length() + b.length() > MAX_LENGTH) throw VMSRuntimeError();
//			else 									 a += b; 
//		}));
//		
//		add_vms<S,S,char>("pair(%s,%s)",  new std::function(+[](MyGrammar::VirtualMachineState_t* vms, int) {
//			char b = vms->getpop<char>();
//			S& a = vms->stack<S>().topref();
//			
//			if(a.length() + 1 > MAX_LENGTH) throw VMSRuntimeError();
//			else 						    a += b; 
//		}));

		add_vms<S,S,S>("append(%s,%s)",  new std::function(+[](MyGrammar::VirtualMachineState_t* vms, int) {
			S b = vms->getpop<S>();
			S a = vms->getpop<S>();
			
			// NOTE: INSTEAD of throwing here, we will just return a. This is different from other models
			// but is needed to deal with the potentially infinite F
			if(a.length() + b.length() > MAX_LENGTH) {
				vms->push(a); // do nothing, leave as-is // throw VMSRuntimeError();
			}
			else {
				vms->push(a+b);
			}
		}));
		add_vms<S,S,char>("pair(%s,%s)",  new std::function(+[](MyGrammar::VirtualMachineState_t* vms, int) {
			S b = vms->getpop<S>();
			auto a = vms->getpop<char>();
			
			// NOTE: INSTEAD of throwing here, we will just return a. This is different from other models
			// but is needed to deal with the potentially infinite F
			if(1 + b.length() > MAX_LENGTH) {
				vms->push(b); // do nothing, leave as-is // throw VMSRuntimeError();
			}
			else {
				vms->push(b+S(1,a));
			}
		}));

		add("\u00D8",        +[]()         -> S          { return S(""); });
		add("eq(%s,%s)",      +[](S x, S y) -> bool       { return x==y; });

		add("and(%s,%s)",    Builtins::And<MyGrammar>);
		add("or(%s,%s)",     Builtins::Or<MyGrammar>);
		add("not(%s)",       Builtins::Not<MyGrammar>);
		
		add("x",               Builtins::X<MyGrammar>);
		add("if_s(%s,%s,%s)",  Builtins::If<MyGrammar,S>);
		add("if_c(%s,%s,%s)",  Builtins::If<MyGrammar,char>);
		add("flip()",          Builtins::Flip<MyGrammar>);
		
		// Standard recursion:
		//add("F(%s)",           Builtins::Recurse<MyGrammar>);
		add("F(%s)",     Builtins::RecurseEmptyOnDepthException<MyGrammar>);
	}
} grammar; 

///~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
/// Declare our hypothesis type
///~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#include "StochasticLOTHypothesis.h"

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
//		prefix = prefix+str(int(getpid()))+"\t"+str(FleetStatistics::global_sample_count)+"\t";
//		prefix += str(grammar->compute_enumeration_order(value)) + "\t"; 
		Super::show(prefix); 
	}
};

#include "Lexicon.h"

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// "Library" reuse prior (see HISTORY.md 2026-07-22)
//
// We maintain a growing library of reusable *statements* (subtrees) harvested from the
// bags processed so far. Scoring a bag, we let it discount any structure it shares with
// something already in the library. Reuse is graded: a statement need not be reproduced
// verbatim to benefit -- the discount is the cost of the largest connected region it
// shares with a stored statement, and multiple disjoint statements can be reused at
// different places in the program.
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

// Connected top-down match of x against a single stored statement s: keep x's rule
// wherever it equals s's aligned rule (recursing into children), and Null (a "gap")
// wherever they differ. The non-Null nodes of the result are exactly the nodes of x that
// s accounts for, as one connected region rooted at x. grammar.log_probability() of this
// result is therefore the prior cost of the reused part of x.
Node connected_match(const Node& x, const Node& s) {
	if(s.rule != x.rule) return grammar.makeNode(NullRule);
	Node out = grammar.makeNode(x.rule);
	for(size_t i=0;i<x.nchildren();i++) {
		Node c = connected_match(x.child(i), s.child(i));
		out.set_child(i, c);
	}
	return out;
}

double library_discount(const Node& x, const std::vector<Node>& library); // fwd decl

// Walk x alongside an anchored match m. Where m left a gap (Null) but x has a real
// subtree there, that subtree is free to reuse a *different* statement -> recurse. Where
// m is present, descend to look for internal gaps deeper down.
double gaps_discount(const Node& x, const Node& m, const std::vector<Node>& library) {
	double d = 0.0;
	for(size_t i=0;i<x.nchildren();i++) {
		if(m.child(i).is_null()) d += library_discount(x.child(i), library);
		else                     d += gaps_discount(x.child(i), m.child(i), library);
	}
	return d;
}

// How many nats of x's prior cost can be discounted by reusing statements from `library`?
// We greedily anchor at x the single stored statement that covers the most (a connected
// region), credit its cost, then recurse into every gap it left -- both the children it
// never reached and the internal holes inside it -- to reuse further, disjoint statements.
// Returns a NON-negative number to add back onto the prior. Because log_probability is
// per-node (context free), the credited nodes are disjoint and their total credit can
// never exceed x's own full cost, so a fully-reused program lands at exactly prior 0.
double library_discount(const Node& x, const std::vector<Node>& library) {
	Node   best    = grammar.makeNode(NullRule);
	double best_lp = 0.0; // an all-Null (empty) match costs 0
	for(const auto& s : library) {
		Node   c  = connected_match(x, s);
		double lp = grammar.log_probability(c);
		if(lp < best_lp) { best_lp = lp; best = c; }
	}

	double discount = -best_lp; // >= 0

	if(best.is_null()) {
		// nothing could anchor at x itself; still try to reuse inside its children
		for(size_t i=0;i<x.nchildren();i++)
			discount += library_discount(x.child(i), library);
	}
	else {
		// x (and part of its subtree) is covered; look for gaps to reuse elsewhere
		discount += gaps_discount(x, best, library);
	}
	return discount;
}

// Add every subtree of x (x itself and all descendants) to the library as reusable
// statements, so later bags can anchor a match against any of them.
void collect_statements(const Node& x, std::vector<Node>& library) {
	for(const auto& n : x) library.push_back(n);
}


struct BagData {
	int bag; // which hypothesis do we run on?
	InnerHypothesis::datum_t datum;
	
	BagData(int b, S o) : bag(b) {
		InnerHypothesis::datum_t x; x.input=""; x.output=o; 
		datum = x;
	}
};

class MyHypothesis final : public Lexicon<MyHypothesis, int,InnerHypothesis, S, S, BagData> {
public:	
	using Super = Lexicon<MyHypothesis, int,InnerHypothesis, S, S, BagData>;
	using Super::Super;
	
	
	// just call this to overwrite
	double compute_single_likelihood(const datum_t& x) override {	
		return this->at(x.bag).compute_single_likelihood(x.datum);
	}
	
	virtual double compute_prior() override {
		// Sequential "library learning" prior. We process bags in order, keeping a library
		// of the statements (subtrees) seen so far. Each bag pays its full grammar prior,
		// minus a graded discount for whatever structure it can reuse from statements
		// already in the library; then its own statements are added to the library for the
		// bags that follow. Bag 0 has an empty library and pays full price. Because the
		// discount can never exceed a program's own cost, a program that fully reuses
		// existing structure lands at prior 0 (free), so an induced overhypothesis
		// propagates forward instead of being rediscovered.
		this->prior = 0.0;

		std::vector<Node> library; // statements harvested from earlier bags (in order)

		for(auto& [k,f] : factors) { // std::map -> ascending key order == bag order
			const double fp = f.compute_prior(); // full grammar prior of this bag's program
			if(fp == -infinity) return this->prior = -infinity;

			if(overlapping and not library.empty()) {
				// pay full price, then credit back the reusable (shared) structure
				this->prior += fp + library_discount(f.get_value(), library);
			}
			else {
				this->prior += fp;
			}

			if(overlapping) collect_statements(f.get_value(), library);
		}

		return this->prior;
	}

	void show(std::string prefix="") override {
		for(auto& [k,v] : factors) {
			print("#", k, v.call("", "<err>").string());
		}
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
	
	// default include to process a bunch of global variables: mcts_steps, mcc_steps, etc
	Fleet fleet("A simple, one-factor formal language learner");
	fleet.add_option("-a,--alphabet", alphabet, "Alphabet we will use"); 	// add my own args
	//fleet.add_option("-d,--data",     datastr, "Comma separated list of input data strings");	
	fleet.initialize(argc, argv);
	
	// since we are only storing the top, we can ignore repeats from MCMC 
	FleetArgs::MCMCYieldOnlyChanges = true;
	
	//------------------
	// Add the terminals to the grammar
	//------------------	
	
	for(const char c : alphabet) {
		grammar.add_terminal( Q(S(1,c)), c);
	}
	
	//------------------
	// set these global variables to make VMS a little faster, less accurate
	//------------------
	
	VirtualMachineControl::MAX_RECURSE = 10;
	VirtualMachineControl::MAX_STEPS = 64; // 256;
	VirtualMachineControl::MAX_OUTPUTS = 256;
	
	//------------------
	// set up the data
	//------------------
	
	// Each bag repeats a SINGLE character. The shared "repeat one letter" recursive
	// structure (F(x):=pair(F(x),c)) is the overhypothesis we hope to induce; only the
	// specific letter differs from bag to bag. Bag lengths need not be equal.
	std::vector<S> targets = {"aaaa", "bbbb", "cccc", "dddd"};
	assert(targets.size() == BAGS);

	// Reveal the data incrementally: walk bag by bag, and within each bag reveal one
	// more character at a time. Already-completed (earlier) bags stay at full length
	// while the current bag grows one character per trial.
	std::vector<std::vector<MyHypothesis::datum_t>> trials;
	for(size_t b=0;b<BAGS;b++){
		for(size_t len=1; len<=targets[b].length(); len++){
			std::vector<MyHypothesis::datum_t> mydata;
			for(size_t k=0;k<b;k++)  mydata.emplace_back(k, targets[k]);            // full earlier bags
			mydata.emplace_back(b, targets[b].substr(0,len));                        // current bag, partial
			trials.push_back(mydata);
		}
	}

	// ------------------
	// 
	// Run
	//------------------
	std::ofstream out("results.tsv");
	out << "data\trank\tposterior\tprior\tlikelihood\thypothesis\tprediction\n";

	for(auto& mydata : trials) {
		// top stores the top hypotheses we have found
		TopN<MyHypothesis> top(1000);

		// Cold start every trial: one random program per bag. No warm-start carry-over,
		// no hand-built hypotheses injected -- just propose and rank by posterior.
		MyHypothesis h0;
		for(size_t b=0;b<BAGS;b++){
			h0.set(b, InnerHypothesis::sample());
		}

		ParallelTempering samp(h0, mydata, FleetArgs::nchains, 1.20);

		for(auto& h : samp.run(Control()) | top | thin(FleetArgs::thin)) {
			UNUSED(h);
		}

		S desc;
		for(auto& d : mydata) desc += str(d.bag) + ":" + d.datum.output + " ";

		// ------------------------------------------------------------------
		// Figure out what we are predicting next, based only on the data seen.
		// Each bag has its own target length (targets[bag]). If the last bag seen
		// is not yet full, we predict the next character *within* it; if it is full,
		// we predict the first character of the next bag. If the very last bag is
		// full, there is nothing left to predict.
		// ------------------------------------------------------------------
		const auto& last = mydata.back();
		int    predict_bag = last.bag;
		size_t predict_pos = last.datum.output.length();
		bool   has_prediction = true;
		if(predict_pos >= targets[predict_bag].length()) { // current bag is full -> next bag
			predict_bag += 1;
			predict_pos  = 0;
			if(predict_bag >= (int)BAGS) has_prediction = false; // all bags finished
		}

		size_t rank = 0;
		for(auto& h : top.sorted(false)) {
			// Evaluate the relevant bag's function and read off the predicted char.
			// argmax() gives the single most probable output string produced by F;
			// the prediction is its character at predict_pos (empty if too short).
			S prediction = "NA"; // no prediction: round is finished
			if(has_prediction) {
				auto dist = h.at(predict_bag).call(S(""), S(""));
				S best = dist.argmax();
				// Filter: drop hypotheses that make no prediction on the next bag, i.e.
				// whose most-probable output is too short to have a char at predict_pos.
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
	}
}

#endif