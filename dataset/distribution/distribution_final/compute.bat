:: Author: Lothar Thiele, ETHZ, 2-2005
::
:: The batch file calls all 
:: processes for a set of runs of
:: selector and variator. It also
:: includes the intiation of processes
:: for the statistical tests.
:: The batch file takes several arguments:
::
::   label arg2 arg3 ... 
::
:: At first, it branches to the label 
:: that is given as an argument
:: in order to do initiate the
:: right processes.

@goto %1

:: This label performs the run of the
:: monitor, selector and variator.
:run
@shift
@cd %1_win
copy ..\%2_win\PISA_cfg PISA_cfg
start /min /b %1.exe %1_param.txt PISA_ 0.1
@cd ../%2_win 
start /min /b %2.exe %2_param.txt PISA_ 0.1
@cd ../monitor_win
monitor.exe ../%2_win/%2_param.txt ../%2_win/PISA_ ../%1_win/%1_param.txt ../%1_win/PISA_ monitor_param.txt ../runs/%2_%1 0.1
@cd ..
@goto end

:: This label computes the collected runs
:: for the test problem (variator), computes the
:: bounds, normalized bounds and reference set.
:bounds
@shift
@cd runs
type %2_*.%3 > ../tests/%2.%3
@cd ../tools_win
bound.exe ../tests/%2.%3 ../tests/%2_bound.%3
normalize.exe ../tests/%2_bound.%3 ../tests/%2.%3 ../tests/%2_norm.%3
filter.exe  ../tests/%2_norm.%3 ../tests/%2_ref.%3
@cd ..
@goto end

:: This label normalizes all runs of selector-variator
:: pairs and determines the indicators.
:indicators
@shift
@cd tools_win
normalize.exe ../tests/%2_bound.%3 ../runs/%2_%1.%3 ../tests/%2_%1_norm.%3
@cd ../indicators_win
hyp_ind.exe ../tests/%2_%1_norm.%3 ../tests/%2_ref.%3 ../tests/%2_%1_hyp.%3
echo. >> ../tests/%2_%1_hyp.%3
eps_ind.exe ../tests/%2_%1_norm.%3 ../tests/%2_ref.%3 ../tests/%2_%1_eps.%3
echo. >> ../tests/%2_%1_eps.%3
r_ind.exe ../tests/%2_%1_norm.%3 ../tests/%2_ref.%3 ../tests/%2_%1_r.%3
echo. >> ../tests/%2_%1_r.%3
@cd..
@goto end

:: This label computes all available statistical
:: tests on all computed indicators
:tests
@shift
@cd tests
type %2_*_eps.%3 > %2_eps.%3
type %2_*_hyp.%3 > %2_hyp.%3
type %2_*_r.%3 > %2_r.%3
@cd ../statistics_win
kruskal-wallis.exe ../tests/%2_eps.%3 kruskalparam.txt ../tests/%2_eps_kruskal.%3
kruskal-wallis.exe ../tests/%2_hyp.%3 kruskalparam.txt ../tests/%2_hyp_kruskal.%3
kruskal-wallis.exe ../tests/%2_r.%3 kruskalparam.txt ../tests/%2_r_kruskal.%3
mann-whit.exe ../tests/%2_eps.%3 emptyparam.txt ../tests/%2_eps_mann.%3
mann-whit.exe ../tests/%2_hyp.%3 emptyparam.txt ../tests/%2_hyp_mann.%3
mann-whit.exe ../tests/%2_r.%3 emptyparam.txt ../tests/%2_r_mann.%3
wilcoxon-sign.exe ../tests/%2_eps.%3 emptyparam.txt ../tests/%2_eps_wilcoxon.%3
wilcoxon-sign.exe ../tests/%2_hyp.%3 emptyparam.txt ../tests/%2_hyp_wilcoxon.%3
wilcoxon-sign.exe ../tests/%2_r.%3 emptyparam.txt ../tests/%2_r_wilcoxon.%3
fisher-matched.exe ../tests/%2_eps.%3 fisherparam.txt ../tests/%2_eps_fisherm.%3
fisher-matched.exe ../tests/%2_hyp.%3 fisherparam.txt ../tests/%2_hyp_fisherm.%3
fisher-matched.exe ../tests/%2_r.%3 fisherparam.txt ../tests/%2_r_fisherm.%3
fisher-indep.exe ../tests/%2_eps.%3 fisherparam.txt ../tests/%2_eps_fisheri.%3
fisher-indep.exe ../tests/%2_hyp.%3 fisherparam.txt ../tests/%2_hyp_fisheri.%3
fisher-indep.exe ../tests/%2_r.%3 fisherparam.txt ../tests/%2_r_fisheri.%3
@cd ..
@goto end

:: This label determines all attainment surfaces 
:: for a given selector-variator pair
:eafs
@shift
@cd attainment_win
eaf.exe -o ../tests/%2_%1_eaf.%3 ../tests/%2_%1_norm.%3
@cd..
@goto end

:: This label compares the attainment surfaces 
:: of two selectors for a given variator
:: using a statistical test
:eaftests
@shift
@if "%1" GEQ "%2" goto end
@cd attainment_win
eaf -i ../tests/tmp ../tests/%3_%1_norm.%4 ../tests/%3_%2_norm.%4 > ../tests/dump
@rm ../tests/dump
eaf-test  ../tests/tmp > ../tests/%3_%1_%2_eaftest.%4
@rm ../tests/tmp
@cd ..
@goto end

:: This label computes the nondominated ranking test for a
:: given variator and a pair of selectors
:ranktests
@shift
@if "%1" GEQ "%2" goto end
@cd indicators_win
dominance-rank.exe ../tests/%3_%1_norm.%4 ../tests/%3_%2_norm.%4 ../tests/dump > ../tests/tmp
@rm ../tests/dump
@cd ../statistics_win
mann-whit.exe ../tests/tmp emptyparam.txt ../tests/%3_%1_%2_ranktest.%4
@rm ../tests/tmp
@cd ..
@goto end

:end
