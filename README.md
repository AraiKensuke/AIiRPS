# AIiRPS

The iterated rock-scissor-paper (iIRPS) game (known as "best of 5!" on school yards) is a strategic game, where players are constantly trying to out-guess each other to gain an advantage, and win.  One possible strategy of play is to be completely random - the Nash mixed equilibrium (NME) strategy, but 1) humans are very bad at being random and 2) such a strategy can't exploit opponents - for example, think playing against a 2 year-old who only plays "rock".  Very exploitable, but playing randomly will only allow you to win 1/3 of the time.  So the random strategy is a strategy that keeps your losses to a minimum, but won't allow you to use what you've learned about your opponent's behavior against them.

So in an iterated setting playing against someone not playing the NME, a possible meta-strategy would be to constantly change the instantaneous strategy - we here mean the rule used to decide whether to play rock, paper or scissors next.  We call this the Next-hand Generating Strategy, and we hypothesize that it is constantly evolving.  One way to study the human social decision-making process would be to have a human player (HP) play against an RPS player that is an artificial intelligence (AI) agent, allowing us to parameterically control how "intelligent" the opponent is, or what general class of algorithm they might use, to learn the HP.  

We provide some sample datasets from HP playing against an AI, and a Google Colab notebook that infers the NGS(k) <A href="https://colab.research.google.com/github/AraiKensuke/AIiRPS/blob/master/Launch.ipynb">here (Run demonstration in Colab)</A>.


