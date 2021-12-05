from environment.config import ScoreBias, ScoreScale
import time

# reference for duplicate bridge scoring rules: https://www.acbl.org/learn_page/how-to-play-bridge/how-to-keep-score/duplicate/
def convert2IMP(init_diff):
	diff = abs(init_diff)
	imp = None
	if diff < 20:
		imp = 0
	elif diff >= 20 and diff < 50:
		imp = 1
	elif diff >= 50 and diff < 90:
		imp = 2
	elif diff >= 90 and diff < 130:
		imp = 3
	elif diff >= 130 and diff < 170:
		imp = 4
	elif diff >= 170 and diff < 220:
		imp = 5
	elif diff >= 220 and diff < 270:
		imp = 6
	elif diff >= 270 and diff < 320:
		imp = 7
	elif diff >= 320 and diff < 370:
		imp = 8
	elif diff >= 370 and diff < 430:
		imp = 9
	elif diff >= 430 and diff < 500:
		imp = 10
	elif diff >= 500 and diff < 600:
		imp = 11
	elif diff >= 600 and diff < 750:
		imp = 12
	elif diff >= 750 and diff < 900:
		imp = 13
	elif diff >= 900 and diff < 1100:
		imp = 14
	elif diff >= 1100 and diff < 1300:
		imp = 15
	elif diff >= 1300 and diff < 1500:
		imp = 16
	elif diff >= 1500 and diff < 1750:
		imp = 17
	elif diff >= 1750 and diff < 2000:
		imp = 18
	elif diff >= 2000 and diff < 2250:
		imp = 19
	elif diff >= 2250 and diff < 2500:
		imp = 20
	elif diff >= 2500 and diff < 3000:
		imp = 21
	elif diff >= 3000 and diff < 3500:
		imp = 22
	elif diff >= 3500 and diff < 4000:
		imp = 23
	elif diff >= 4000:
		imp = 24
	if init_diff >= 0:
		return imp
	return -imp

# reference for duplicate bridge scoring rules: https://www.acbl.org/learn_page/how-to-play-bridge/how-to-keep-score/duplicate/
def score(input_tuple):
	# bid_tricks range from 1 - 7
	# max_tricks range from 0 - 13
	bid_tricks, trump, max_tricks, vulner, is_double = input_tuple
	declarer_score = 0

	# Major suit trump.
	# Suit2str = {0: "S", 1: "H", 2: "D", 3: "C"}
	if (bid_tricks+6) <= max_tricks: #make the contract
		# contract score
		factor = 2**is_double
		declarer_score += bid_tricks*ScoreScale[trump]*factor
		declarer_score += ScoreBias[trump]*factor
		# game bonus
		if declarer_score >= 100:
			if vulner == 0:
				declarer_score += 300
			else:
				declarer_score += 500
		else:
			declarer_score += 50
		# slam bonus
		if bid_tricks == 6:
			if vulner == 0:
				declarer_score += 500
			else:
				declarer_score += 750
		elif bid_tricks == 7:
			if vulner == 0:
				declarer_score += 1000
			else:
				declarer_score += 1500
		# double bonus
		if is_double == 1:
			declarer_score += 50
		elif is_double == 2:
			declarer_score += 100
		# overtrick score
		over_trick = max_tricks - (bid_tricks + 6)
		if vulner == 0:
			if is_double == 0:
				declarer_score += over_trick * ScoreScale[trump]
			elif is_double == 1:
				declarer_score += over_trick * 100
			elif is_double == 2:
				declarer_score += over_trick * 200
		elif vulner == 1:
			if is_double == 0:
				declarer_score += over_trick * ScoreScale[trump]
			elif is_double == 1:
				declarer_score += over_trick * 200
			elif is_double == 2:
				declarer_score += over_trick * 400
	else:
		# down score
		under_trick = bid_tricks + 6 - max_tricks
		if vulner == 0:
			if is_double == 0:
				declarer_score += under_trick * 50
			else:
				factor = 2**(is_double-1)
				if under_trick == 1:
					declarer_score += 100 * factor
				elif under_trick == 2:
					declarer_score += 300 * factor
				elif under_trick == 3:
					declarer_score += 500 * factor
				else:
					declarer_score += (500 + (under_trick - 3) * 300) * factor
		elif vulner == 1:
			if is_double == 0:
				declarer_score += under_trick * 100
			else:
				factor = 2**(is_double-1)
				if under_trick == 1:
					declarer_score += 200 * factor
				else:
					declarer_score += (200 + (under_trick - 1) * 300) * factor

		declarer_score = -declarer_score

	return declarer_score


# calculate the table
def precompute_scores():
	input_space = [(bid_tricks, trump, max_tricks, vulner, is_double)  for bid_tricks in range(1,8)
				   									for trump in range(5)
				   									for max_tricks in range(14)
				   									for vulner in range(2)
				   									for is_double in range(3)]

	score_space = [score(t) for t in input_space]
	scorer = dict(zip(input_space, score_space))
	del input_space, score_space
	return scorer

start_t  = time.time()
print("Calculating SCORE_TABLE")
SCORE_TABLE = precompute_scores()
print("SCORE_TABLE CALCULATION COMPLETED")
end_t = time.time()
print("The total time use is {}.".format(end_t-start_t))

# check according to the scoring sheet: http://web2.acbl.org/documentLibrary/play/InstantScorer.pdf
# for key in SCORE_TABLE:
# 	bid_tricks, trump, max_tricks, vulner, is_double = key
	# if (bid_tricks + 6 - max_tricks) == 13:
	# 	print(key, ": ", SCORE_TABLE[key])
	# if bid_tricks==7 and trump==4 and (max_tricks - 6 - bid_tricks)>=0:
		# tup = (bid_tricks, 3, max_tricks, vulner, is_double)
		# assert SCORE_TABLE[tup]==SCORE_TABLE[key], SCORE_TABLE[tup]
		# print(key, ": ", SCORE_TABLE[key])
		# print(tup, ": ", SCORE_TABLE[tup])