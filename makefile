help:
	@echo -e "\tmake help - display this help message!"
	@echo -e "\tmake stegoalgs - download external stegonagraphic algorithms"
	@echo -e "\tmake boss - get the boss dataset"

all: help


stegoalgs: external/s-uniward external/wow

boss: data/raw/boss


external: 
	mkdir external

data:
	mkdir data

data/raw: data
	mkdir data/raw


external/s-uniward: external
	mkdir external/s-uniward
	wget http://dde.binghamton.edu/download/stego_algorithms/download/S-UNIWARD_linux_make_v10.tar.gz
	mv S-UNIWARD_linux_make_v10.tar.gz external/s-uniward
	cd external/s-uniward && tar xzvf S-UNIWARD_linux_make_v10.tar.gz 
	rm external/s-uniward/S-UNIWARD_linux_make_v10.tar.gz 

external/wow: external
	mkdir external/wow
	wget http://dde.binghamton.edu/download/stego_algorithms/download/WOW_linux_make_v10.tar.gz
	mv WOW_linux_make_v10.tar.gz external/wow
	cd external/wow && tar xzvf WOW_linux_make_v10.tar.gz 
	rm external/wow/WOW_linux_make_v10.tar.gz 
	
data/raw/boss: data/raw
	mkdir data/raw/boss
	cd data/raw/boss; \
		wget http://agents.fel.cvut.cz/stegodata/PGMs/BossBase-1.01-hugo-alpha=0.4.tar.bz2; \
		bunzip2 BossBase-1.01-hugo-alpha=0.4.tar.bz2; \
		mkdir stego; \
		tar xf BossBase-1.01-hugo-alpha=0.4.tar -C stego; \
		wget http://agents.fel.cvut.cz/stegodata/PGMs/BossBase-1.01-cover.tar.bz2; \
		bunzip2 BossBase-1.01-cover.tar.bz2; \
		mkdir cover; \
		tar xf BossBase-1.01-cover.tar -C cover
	
	# rm BossBase-1.01-hugo-alpha=0.4.tar.bz2
	# rm BossBase-1.01-cover.tar.bz2
.PHONY: help stegoalgs boss
