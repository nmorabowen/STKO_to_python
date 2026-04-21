
recorder mpco "results.mpco" \
-N "displacement" "rotation" "velocity" "angularVelocity" "acceleration" "angularAcceleration" "reactionForce" "reactionMoment" "reactionForceIncludingInertia" "reactionMomentIncludingInertia" "rayleighForce" "rayleighMoment" "unbalancedForce" "unbalancedMoment" "unbalancedForceIncludingInertia" "unbalancedMomentIncludingInertia" "pressure" "modesOfVibration" "modesOfVibrationRotational" \
-E "force" "deformation" "localForce" "damage" "equivalentPlasticStrain" "cw" "section.force" "section.deformation" "material.stress" "material.strain" "material.damage" "material.equivalentPlasticStrain" "material.cw" "section.fiber.stress" "section.fiber.strain" "section.fiber.damage" "section.fiber.equivalentPlasticStrain" "section.fiber.cw"

# Constraints.sp fix
	fix 1 1 1 1 1 1 1
	fix 3 1 1 1 1 1 1

# Patterns.addPattern loadPattern
pattern Plain 3 1 {

# Loads.Force EdgeForce
	load 4 0.0 0.0 -25000.0 0.0 0.0 0.0
	load 2 0.0 0.0 -25000.0 0.0 0.0 0.0
}

# analyses command
domainChange
constraints Transformation
numberer RCM
system UmfPack
test NormUnbalance 0.0001 10  
algorithm Linear
integrator LoadControl 0.0
analysis Static
# ======================================================================================
# NON-ADAPTIVE LOAD CONTROL ANALYSIS
# ======================================================================================

# ======================================================================================
# USER INPUT DATA 
# ======================================================================================

# duration and initial time step
set total_duration 1.0
set initial_num_incr 10

set STKO_VAR_time 0.0
set STKO_VAR_time_increment [expr $total_duration / $initial_num_incr]
set STKO_VAR_initial_time_increment $STKO_VAR_time_increment
integrator LoadControl $STKO_VAR_time_increment 
for {set STKO_VAR_increment 1} {$STKO_VAR_increment <= $initial_num_incr} {incr STKO_VAR_increment} {
	
	# before analyze
	STKO_CALL_OnBeforeAnalyze
	
	# perform this step
	set STKO_VAR_analyze_done [analyze 1 ]
	
	# update common variables
	if {$STKO_VAR_analyze_done == 0} {
		set STKO_VAR_num_iter [testIter]
		set STKO_VAR_time [expr $STKO_VAR_time + $STKO_VAR_time_increment]
		set STKO_VAR_percentage [expr $STKO_VAR_time/$total_duration]
		set norms [testNorms]
		if {$STKO_VAR_num_iter > 0} {set STKO_VAR_error_norm [lindex $norms [expr $STKO_VAR_num_iter-1]]} else {set STKO_VAR_error_norm 0.0}
	}
	
	# after analyze
	set STKO_VAR_afterAnalyze_done 0
	STKO_CALL_OnAfterAnalyze
	
	# check convergence
	if {$STKO_VAR_analyze_done == 0} {
		# print statistics
		if {$STKO_VAR_process_id == 0} {
			puts [format "Increment: %6d | Iterations: %4d | Norm: %8.3e | Progress: %7.3f %%" $STKO_VAR_increment $STKO_VAR_num_iter  $STKO_VAR_error_norm [expr $STKO_VAR_percentage*100.0]]
		}
	} else {
		# stop analysis
		error "ERROR: the analysis did not converge"
	}
	
}

# done
if {$STKO_VAR_process_id == 0} {
	puts "Target time has been reached. Current time = $STKO_VAR_time"
	puts "SUCCESS."
}

loadConst
wipeAnalysis

# Patterns.addPattern loadPattern
pattern Plain 5 1 {

# Loads.Force NodeForce
	load 4 10000.0 0.0 0.0 0.0 0.0 0.0
}

# analyses command
domainChange
constraints Transformation
numberer RCM
system UmfPack
test NormUnbalance 0.0001 10  
algorithm Linear
integrator LoadControl 0.0
analysis Static
# ======================================================================================
# NON-ADAPTIVE LOAD CONTROL ANALYSIS
# ======================================================================================

# ======================================================================================
# USER INPUT DATA 
# ======================================================================================

# duration and initial time step
set total_duration 1.0
set initial_num_incr 10

set STKO_VAR_time 0.0
set STKO_VAR_time_increment [expr $total_duration / $initial_num_incr]
set STKO_VAR_initial_time_increment $STKO_VAR_time_increment
integrator LoadControl $STKO_VAR_time_increment 
for {set STKO_VAR_increment 1} {$STKO_VAR_increment <= $initial_num_incr} {incr STKO_VAR_increment} {
	
	# before analyze
	STKO_CALL_OnBeforeAnalyze
	
	# perform this step
	set STKO_VAR_analyze_done [analyze 1 ]
	
	# update common variables
	if {$STKO_VAR_analyze_done == 0} {
		set STKO_VAR_num_iter [testIter]
		set STKO_VAR_time [expr $STKO_VAR_time + $STKO_VAR_time_increment]
		set STKO_VAR_percentage [expr $STKO_VAR_time/$total_duration]
		set norms [testNorms]
		if {$STKO_VAR_num_iter > 0} {set STKO_VAR_error_norm [lindex $norms [expr $STKO_VAR_num_iter-1]]} else {set STKO_VAR_error_norm 0.0}
	}
	
	# after analyze
	set STKO_VAR_afterAnalyze_done 0
	STKO_CALL_OnAfterAnalyze
	
	# check convergence
	if {$STKO_VAR_analyze_done == 0} {
		# print statistics
		if {$STKO_VAR_process_id == 0} {
			puts [format "Increment: %6d | Iterations: %4d | Norm: %8.3e | Progress: %7.3f %%" $STKO_VAR_increment $STKO_VAR_num_iter  $STKO_VAR_error_norm [expr $STKO_VAR_percentage*100.0]]
		}
	} else {
		# stop analysis
		error "ERROR: the analysis did not converge"
	}
	
}

# done
if {$STKO_VAR_process_id == 0} {
	puts "Target time has been reached. Current time = $STKO_VAR_time"
	puts "SUCCESS."
}

loadConst
wipeAnalysis

# Done!
puts "ANALYSIS SUCCESSFULLY FINISHED"
