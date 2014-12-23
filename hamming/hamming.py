def get_parity(databits):
	parity = 0
	bitvector = 3  # binary representation of bit no. in hamming algorithm,
	               # i.e. 3rd bit: 011, 4th bit: 100, 5th bit: 101, etc.
	while databits > 0:
		if not (bitvector & (bitvector - 1)):  # check if bitvector is power of 2
			bitvector += 1
		if databits & 1:                 # if lsb is one
			parity = parity ^ bitvector  # twiddle parity bits correctly
		databits = databits >> 1
		bitvector += 1
	return bin(parity)
