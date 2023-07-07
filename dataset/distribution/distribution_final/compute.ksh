#! /usr/bin/ksh
#
# compute.ksh: UNIX/LINUX complement to compute.bat
#
# Eckart Zitzler, ETH Zurich, Feb 18, 2005

#=================================================
# DO NOT MAKE ANY CHANGES BELOW THIS LINE
#=================================================

case $1 in
    run)
	cd $2_$os ;
	cp ../$3_$os/PISA_cfg . ;
	./$2 $2_param.txt PISA_ 0.1 &
	cd ../$3_$os ;
	./$3 $3_param.txt PISA_ 0.1 &
	cd ../monitor_$os
	./monitor ../$3_$os/$3_param.txt ../$3_$os/PISA_ ../$2_$os/$2_param.txt ../$2_$os/PISA_ monitor_param.txt ../runs/$3_$2 0.1 ;
	cd ..
	;;
    bounds) 
	cd runs ;
	cat $3_*.$4 > ../tests/$3.$4 ;
	cd ../tools_$os ;
	./bound ../tests/$3.$4 ../tests/$3_bound.$4 ;
	./normalize ../tests/$3_bound.$4 ../tests/$3.$4 ../tests/$3_norm.$4 ;
	./filter  ../tests/$3_norm.$4 ../tests/$3_ref.$4 ;
	cd ..
	;;
    indicators)
	cd tools_$os ;
	./normalize ../tests/$3_bound.$4 ../runs/$3_$2.$4 ../tests/$3_$2_norm.$4 ;
	cd ../indicators_$os ;
	./hyp_ind ../tests/$3_$2_norm.$4 ../tests/$3_ref.$4 ../tests/$3_$2_hyp.$4 ;
	echo >> ../tests/$3_$2_hyp.$4 ;
	./eps_ind ../tests/$3_$2_norm.$4 ../tests/$3_ref.$4 ../tests/$3_$2_eps.$4 ;
	echo >> ../tests/$3_$2_eps.$4 ;
	./r_ind ../tests/$3_$2_norm.$4 ../tests/$3_ref.$4 ../tests/$3_$2_r.$4 ;
	echo >> ../tests/$3_$2_r.$4 ;
	cd ..
	;;
    tests)
	cd tests ;
	cat $3_*_eps.$4 > $3_eps.$4 ;
	cat $3_*_hyp.$4 > $3_hyp.$4 ;
	cat $3_*_r.$4 > $3_r.$4 ;
	cd ../statistics_$os;
	./kruskal-wallis ../tests/$3_eps.$4 kruskalparam.txt ../tests/$3_eps_kruskal.$4 ;
	./kruskal-wallis ../tests/$3_hyp.$4 kruskalparam.txt ../tests/$3_hyp_kruskal.$4 ;
	./kruskal-wallis ../tests/$3_r.$4 kruskalparam.txt ../tests/$3_r_kruskal.$4 ;
	./mann-whit ../tests/$3_eps.$4 emptyparam.txt ../tests/$3_eps_mann.$4 ;
	./mann-whit ../tests/$3_hyp.$4 emptyparam.txt ../tests/$3_hyp_mann.$4 ;
	./mann-whit ../tests/$3_r.$4 emptyparam.txt ../tests/$3_r_mann.$4 ;
        ./wilcoxon-sign ../tests/$3_eps.$4 emptyparam.txt ../tests/$3_eps_wilcoxon.$4 ;
        ./wilcoxon-sign ../tests/$3_hyp.$4 emptyparam.txt ../tests/$3_hyp_wilcoxon.$4 ;
        ./wilcoxon-sign ../tests/$3_r.$4 emptyparam.txt ../tests/$3_r_wilcoxon.$4 ;
        ./fisher-matched ../tests/$3_eps.$4 fisherparam.txt ../tests/$3_eps_fisherm.$4 ;
        ./fisher-matched ../tests/$3_hyp.$4 fisherparam.txt ../tests/$3_hyp_fisherm.$4 ;
        ./fisher-matched ../tests/$3_r.$4 fisherparam.txt ../tests/$3_r_fisherm.$4 
        ./fisher-indep ../tests/$3_eps.$4 fisherparam.txt ../tests/$3_eps_fisheri.$4 ;
        ./fisher-indep ../tests/$3_hyp.$4 fisherparam.txt ../tests/$3_hyp_fisheri.$4 ;
        ./fisher-indep ../tests/$3_r.$4 fisherparam.txt ../tests/$3_r_fisheri.$4
	cd ..
	;;
    eafs)
        cd attainment_$os ;
	./eaf -o ../tests/$3_$2_eaf.$4 ../tests/$3_$2_norm.$4 ;
	cd ..
	;;
    eaftests)
    if [[ $2 < $3 ]] ; then 
        cd attainment_$os ;
	./eaf -i ../tests/tmp ../tests/$4_$2_norm.$5 ../tests/$4_$3_norm.$5 > ../tests/dump ;
	./eaf-test ../tests/tmp > ../tests/$4_$2_$3_eaftest.$5 ;
	rm ../tests/dump ;
	rm ../tests/tmp ;
	cd .. ;
    fi
	;;
    ranktests)
    if [[ $2 < $3 ]] ; then 
    	cd indicators_$os ;
	./dominance-rank ../tests/$4_$2_norm.$5 ../tests/$4_$3_norm.$5 ../tests/dump > ../tests/tmp ;
	cd ../statistics_$os ;
	./mann-whit ../tests/tmp emptyparam.txt ../tests/$4_$2_$3_ranktest.$5 ;
	rm ../tests/dump ;
	rm ../tests/tmp ;
	cd .. ;
    fi
    ;;
esac
