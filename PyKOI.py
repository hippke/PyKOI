# Tested with Python 3.5

"""
To be done:
- change filename to masterdata.csv and upload
- check if LC SC mixture caused by MAST or corrupt csv files
- Test running mean for LC and SC
"""


import kplr
# A Python interface to the Kepler data, http://dan.iel.fm/kplr/
# To install, use "pip install kplr"

import numpy
import math
import matplotlib.pyplot as plt
import matplotlib.patches as patches


def get_max(phase, flux, start, end):
    """Returns maximum flux value inside given part of phase segment"""
    return numpy.max(flux[numpy.where((start < phase) & (phase < end))])


def weighted_average(phase, flux, weights, window_size):
    """Takes timeflux data with weights per point, returns sliding weighted mean
    A sliding mean (or median) without weights would be much faster to calculate
    """

    # We save the mid window point and mean flux in each window using a list
    # (instead of a numpy array), because it is much faster.
    # We move one step forward after each window. For very large datasets, this
    # can take several minutes, and could be changed to a step size of e.g. 10
    mid_window_times = []
    mean_values = []

    # The number of steps is the whole phase space minus one window size
    steps = (numpy.size(phase) - window_size)

    # The numpy "average" method fails for a weight that is zero.
    # We got weights (x) from standard deviations and have to use them as 1 / x
    # For missing data, weights can be zero
    # To avoid division by zero, we first convert all "0" to "inf"
    weights[weights == 0] = numpy.inf

    for box_start in range(int(steps)):
        start_index = int(box_start)
        end_index = start_index + window_size
        midpoint = numpy.average(phase[start_index:end_index])
        mid_window_times.append(midpoint)
        avg = numpy.ma.average(
            flux[start_index:end_index],
            weights=(1 / weights[start_index:end_index]))
        mean_values.append(avg)

    return mid_window_times, mean_values


def sigma_clipper(time, flux, sigma_clip):
    """Clips all values with flux above n sigma
    Sometimes Kepler gets a cosmic ray hit on a pixel resulting in crazy high
    flux values. Therefore we clip all values above a certain threshold
    I recommend to set a value so that statistically we expect to discard
    only a few values, e.g. 3..6 sigma, resulting in clipping of 0..5 values.
    We shall always check how many are really discarded, as this often shows
    various types of errors in the processing"""
    
    standard_deviation = numpy.std(flux)
    average_flux = numpy.mean(flux)
    upper_clip = average_flux + sigma_clip * standard_deviation
    select = numpy.where(flux < upper_clip)
    cleaned_flux = flux[select]
    cleaned_time = time[select]
    number_of_removed_values = numpy.size(flux) - numpy.size(cleaned_flux)
    print('Clipping ceiling:', upper_clip, 'values removed:', number_of_removed_values)

    return cleaned_time, cleaned_flux

    
def remove_LC_data(time, flux):
    """Returns data without LC part, if mixed SC and LC given"""
    
    # Identification by time difference between subsequent (sorted) points, 
    # which for LC is 0.02042853 to 0.02044024 days
    # (the variation is due to Kepler's motion around the sun --> BJD converted)
    # Of course, if SC data occurs to have a random gap in the data with exactly
    # that duration, one SC value gets incorrectly removed. Tests showed that 
    # this happens extremly rarely by chance and can safely be accepted.
    min_LC_gap = 0.0204285  # round down
    max_LC_gap = 0.0204403  # round up
    original_number_of_values = numpy.size(time)

    # We iterate over multiples of LC gap lengths
    # Tests showed that multiples = 100 catch >99.99% of all LC values
    # The remaining weight of a few LC values amongst one million SC values is
    # very close to zero and can be accapted.
    # We use boolean OR (|) which is overloaded by numpy
    
    multiples = 100
    for gap_multiple in range(1, multiples):
        gap_times = numpy.diff(time)
        select_SC_data = numpy.where(
            (gap_times < min_LC_gap * gap_multiple) | (gap_times > max_LC_gap * gap_multiple))
        time = time[select_SC_data]
        flux = flux[select_SC_data]

    final_number_of_values = numpy.size(time)
    number_of_LC_values_removed = original_number_of_values - final_number_of_values
    print('Removed', number_of_LC_values_removed, 'LC values')

        
    return time, flux


def get_lightcurve_from_file(koi_name):
    """Reads the data from a previously saved KOI csv file"""
    koi = numpy.genfromtxt(path + koi_name + '_datafile.csv',
        dtype=[('time', 'f8'), ('flux', 'f8')])

    return koi['time'], koi['flux']


def get_lightcurve_from_mast(koi_name, use_short_cadence, data_type):
    """Pulls lightcurve from MAST and returns time + flux"""

    client = kplr.API()
    koi = client.koi(koi_name)
    lightcurves = koi.get_light_curves(short_cadence=use_short_cadence)

    time_array = numpy.array([])
    flux_array = numpy.array([])

    for lightcurve in lightcurves:
        time_segment_cleaned = numpy.array([])
        flux_segment_cleaned = numpy.array([])
        with lightcurve.open() as file:
            current_time_segment = file[1].data["time"]
            current_flux_segment = file[1].data[data_type]

        # Select only data that are NOT "NaN" (= missing)
        select_valid_data = numpy.invert(numpy.isnan(current_flux_segment))
        time_segment_cleaned = current_time_segment[select_valid_data]
        flux_segment_cleaned = current_flux_segment[select_valid_data]

        # Normalize each quarter by its median
        flux_segment_cleaned = numpy.divide(
            flux_segment_cleaned, numpy.median(flux_segment_cleaned))
        time_array = numpy.append(time_array, time_segment_cleaned)
        flux_array = numpy.append(flux_array, flux_segment_cleaned)

    # Convert from "Kepler" BJD time to "Holczer 2016+" BJD time (offsets..grrr)
    time_array = time_array + 2454833 - 2454900

    return cleaned_time, cleaned_flux


def extract_segment(time, flux, mid_transit, transit_duration, segment_half_width,
    polynom_degree, make_detrending_figures, blinding, sigma_clip_after_detrending):
    """Extracts a flux + time segment from the total stream,
       segment_half_width is in transit durations"""

    return_empty_array = False  # This is the default case.
    # If something goes wrong, we will set it to True and return empty

    """
    # Planets with very short transit durations suffer from the smear effect
    # (2010MNRAS.408.1758K). Part of the transit flux loss is smeared outside
    # of the actual transit window, because the transit duration is a
    # significant part of the cadence. We can compensate for this by extending
    # the calculated transit window by up to one cadence, so that all transit
    # flux falls into the transit window.
    
    # As this feature is currently not complete, we do not expose the method
    do_smear_correction = False
    if do_smear_correction:
        # Check smallest time between two points to determine whether LC or SC data
        # This does not work correctly yet for cases of mixed SC and LC data
        # Will have to implement method to measure for every single transit,
        # in case we want to mix SC and LC data
        smallest_gap = numpy.min(numpy.diff(numpy.sort(time))) * 24 * 60
        if smallest_gap > 25:
            smear_compensation = 30 / 24 / 60  # 1 LC cadence
            transit_duration = transit_duration + smear_compensation
            print('LC data, smear adjusted transit duration to', transit_duration)
        else:
            smear_compensation = 0
            print('SC data, no smear compensation', transit_duration)
    """
            
    buffered_transit_start = mid_transit - blinding * transit_duration
    buffered_transit_end = mid_transit + blinding * transit_duration
    segment_start = mid_transit - segment_half_width * transit_duration
    segment_end = mid_transit + segment_half_width * transit_duration

    # Select in-transit flux and phase
    select_intransit = numpy.where(
        (buffered_transit_start < time) & (time < buffered_transit_end))
    intransit_time_segment = time[select_intransit]
    intransit_flux_segment = flux[select_intransit]

    # Select out of transit flux and phase
    # We use boolean AND (&) with OR (|) which is overloaded by numpy
    select_outoftransit = numpy.where(
        (segment_start < time) & (time < buffered_transit_start) |
        (buffered_transit_end < time) & (time < segment_end))
    outoftransit_time_segment = time[select_outoftransit]
    outoftransit_flux_segment = flux[select_outoftransit]

    # Check if we have sufficient data for detrending, otherwise the polynomial 
    # fit will may the dust. Shouldn't happen, but when detrending millions of
    # segments, it occured...
    if numpy.size(outoftransit_flux_segment) > (polynom_degree + 1):

        # Fit a polynomial for detrending to the out-of-transit (+buffer) part
        poly = numpy.polyfit(
            outoftransit_time_segment,
            outoftransit_flux_segment,
            polynom_degree)
        polynom = numpy.poly1d(poly)

        # Combine in- and out of transit flux segments to divide both by polynom
        allflux = numpy.append(intransit_flux_segment, outoftransit_flux_segment)
        alltime = numpy.append(intransit_time_segment, outoftransit_time_segment)
        detrended_flux = allflux - polynom(alltime) + 1
        
        # Sigma-clip outlier data again after detrending, can be more agressive
        # now. This also covers only positive values, because we want to protect 
        # the actual transit. Check consistency with zero clipping!
        alltime, detrended_flux = sigma_clipper(
            alltime,
            detrended_flux,
            sigma_clip=sigma_clip_after_detrending)

        # Make figure to check polynomial fit
        if make_detrending_figures:
            print('Making figure...')
            xp = numpy.linspace(segment_start, segment_end, 1000)
            plt.scatter(intransit_time_segment, intransit_flux_segment, color='blue')
            plt.scatter(outoftransit_time_segment, outoftransit_flux_segment, color='red')
            plt.scatter(alltime, detrended_flux - 0.01, color='black')
            plt.plot(xp, polynom(xp), color='black', linestyle='dashed')
            plt.xlabel('Time (days)')
            plt.ylabel('Flux')
            plt.xlim([segment_start, segment_end])
            filename = str(int(mid_transit)) + '.png'
            plt.savefig(path + filename, bbox_inches='tight')
            plt.show()
            plt.clf()

        # Check if a gap >1 hour (2 LC cadences) is present in current segment
        # If yes, print a notice and return empty result arrays
        # To find the largest gap, we sort the time array, take the difference
        # between each dorted value, and finally take the largest difference
        try:
            largest_gap = numpy.max(numpy.diff(numpy.sort(alltime)))
            if largest_gap < 0.05:  # unit: days. 2 LCs are ~0.042 days
                # Convert time axis (in days) to phase axis (in transit durations)
                phase = (alltime - mid_transit) / transit_duration
            else:
                print('Droping transit, too large gap', largest_gap * 60 * 24, 'minutes')
                return_empty_array = True
        except:
            print('All values got sigma-clipped, discarding transit')
            return_empty_array = True
        
    # Other case: We have not enough data
    else:
        return_empty_array = True

    # Something went wrong --> return empty array
    if return_empty_array:
        print('This transfer segment failed')
        phase = numpy.array([])
        detrended_flux = numpy.array([])
        outoftransit_scatter = numpy.array([])

    return phase, detrended_flux, numpy.std(outoftransit_flux_segment)

    
def load_master_data_catalog(filename):
    """Returns master data catalog from file (merged from Holczer 2016+)"""
    catalog = numpy.genfromtxt(
    filename,
    skip_header=1,
    delimiter='\t',
    dtype=[
        ('koi', 'U12'),  # KOI number, e.g. "1300.01" for KOI 1300.01
        ('transit_number', 'int'),  # not always in ascending order
        ('period', 'f8'),  # planetary period in days
        ('average_transit_duration', 'f8'),  # planetary TD in hours
        ('mid_transit_linear', 'f8'),  # linear ephemeris in BJD (days)
        ('TTV', 'f8'),  # in deviations from linear ephemeris, in minutes
        ('TDV', 'f8'),  # in fractions of average TD
        ('flag', 'bool')])  # "True": A sister planet transit occurs nearby

    # The ("flag" = True) indicates that a sister planet transit occurs within 
    # 5 transit durations from the TTV-adjusted mid-transit time.
    # Holczer 2016+ only gives collissions, so this flag has been calculated
    # with a different script, and results have been been manually verified.
    # I recommend to discard affected transits as they can offset the detrending.

    return catalog


def get_koi_phasefold(koi_name, masterdata, sigma_clip_before_detrending,
    sigma_clip_after_detrending, use_max_ttv, make_detrending_figures, blinding,
    use_ttv, segment_half_width, polynom_degree, use_long_cadence):
    """Returns the phase-folded, detrended transit segments for one KOI"""
    
    print('Getting data...')
    time, flux = get_lightcurve_from_file(koi_name)
    #time, flux = get_lightcurve_from_mast(koi_name, use_short_cadence, data_type)
    
    # Remove LC data if desired
    if not use_long_cadence:
        print('Removing LC data...')
        time, flux = remove_LC_data(time, flux)
    
    # Clips all values with flux above n sigma (for "why" see function)
    print('Clipping with sigma=', sigma_clip_before_detrending)
    time, flux = sigma_clipper(time, flux, sigma_clip_before_detrending)
    
    this_koi_flux = numpy.array([])
    this_koi_phase = numpy.array([])
    this_koi_weights = numpy.array([])

    # Iterate over catalog and take the segments for all transits of this KOI
    for row in range(len(masterdata['koi'])):

        # We select the transits for this KOI which are not flagged (see definition)
        if masterdata['koi'][row] == koi_name and not masterdata['flag'][row]:

            # Linear ephemeris by Holczer 2016+ in BJD (days)
            T_linear = masterdata['mid_transit_linear'][row]

            # TTVs are given by Holczer 2016+ as deviations from the linear ephemeris
            ttv_minutes = masterdata['TTV'][row]
            ttv_days = ttv_minutes / 60 / 24
            if use_ttv:
                Tx = T_linear + ttv_days
            else:
                Tx = T_linear

            # TDVs are given by Holczer 2016+ as fractions of the average TD
            # In many cases, they are zero
            average_transit_duration_hours = masterdata['average_transit_duration'][row]
            average_transit_duration_days = average_transit_duration_hours / 24
            TDV = masterdata['TDV'][row]
            current_transit_duration = average_transit_duration_days * (1 + TDV)

            # If the current TTV is below the desired upper limit,
            # fetch detrended segment and append it to the total KOI data
            if abs(ttv_minutes) < use_max_ttv:
                segment_phase, segment_flux, outoftransit_scatter = extract_segment(
                    time,
                    flux,
                    mid_transit=Tx,
                    transit_duration=current_transit_duration,
                    segment_half_width=segment_half_width,
                    polynom_degree=polynom_degree,
                    make_detrending_figures=make_detrending_figures,
                    blinding=blinding,
                    sigma_clip_after_detrending=sigma_clip_after_detrending)
                print('Working on transit at T=', Tx)
                std_array = numpy.zeros(numpy.size(segment_phase))
                std_array.fill(outoftransit_scatter)
                this_koi_flux = numpy.append(this_koi_flux, segment_flux)
                this_koi_phase = numpy.append(this_koi_phase, segment_phase)
                this_koi_weights = numpy.append(this_koi_weights, std_array)
            else:
                print('Too large TTV for transit at T=', Tx, 'with TTV=',
                    abs(ttv_minutes), '>', use_max_ttv, 'min')

    # Now we have all data for the KOI and have to sort it by phase
    print('Sorting', numpy.size(this_koi_flux), 'values...')
    stacked_data = numpy.column_stack((this_koi_phase, this_koi_flux, this_koi_weights))
    sorted_data = stacked_data[stacked_data[:, 0].argsort()]
    this_koi_phase = sorted_data[:,0]
    this_koi_flux = sorted_data[:,1]
    this_koi_weights = sorted_data[:,2]                
                    
    return this_koi_phase, this_koi_flux, this_koi_weights


def make_figure(phase, flux, mid_window_times, mean_values, window_size,
    segment_half_width, blinding, polynom_degree, use_max_ttv, use_ttv,
    peak_left, reference_left, peak_right, reference_right):

    ax = plt.gca()
    
    # Actual data
    plt.plot(mid_window_times, mean_values, color='black', linewidth=2)
    plt.scatter(phase, flux, alpha=0.2, s=2, color='blue')

    # Do not use exponential notation
    ax.get_yaxis().get_major_formatter().set_useOffset(False)
    ax.get_yaxis().get_major_formatter().set_scientific(False)
    
    # Vertical lines for transit and blinding area 
    plt.plot([-0.5,-0.5], [-1000, 2000], color='gray')
    plt.plot([+0.5,+0.5], [-1000, 2000], color='gray')
    plt.plot([-5, 5], [0, 0], color='gray')
    plt.plot([-blinding,-blinding], [-1000, 2000], color='gray', \
        linestyle = 'dashed')
    plt.plot([+blinding,+blinding], [-1000, 2000], color='gray', \
        linestyle = 'dashed')

    # Rectangle to show window size of weighted runnning mean
    ax.add_patch(patches.Rectangle((-segment_half_width + 0.2, -1000), \
        window_size, 2000, fill=True, alpha=0.5, color='red'))
        
    # Annotations
    plt.annotate('segment width = ' + str(segment_half_width) + ' TDs', \
        xy=(-segment_half_width + 0.1, 440), backgroundcolor='white')
    plt.annotate('polynom degree = ' + str(polynom_degree), \
        xy=(-segment_half_width + 0.1, 380), backgroundcolor='white')
    plt.annotate('max allowed TTV = ' + str(use_max_ttv) + ' min', \
        xy=(-segment_half_width + 0.1, 320), backgroundcolor='white')
    plt.annotate('use TTV = ' + str(use_ttv), \
        xy=(-segment_half_width + 0.1, 260), backgroundcolor='white')
    plt.annotate('mask size = ' + str(blinding * 2) + ' TDs',\
        xy=(-segment_half_width + 0.1, 200), backgroundcolor='white')
    plt.annotate('left ' + str(int(peak_left)) + '±' + str(int(reference_left)) +\
        ' ppm', xy=(segment_half_width - 1.8, -530), backgroundcolor='white')
    plt.annotate('right ' + str(int(peak_right)) + '±' + str(int(reference_right)) \
        + ' ppm', xy=(segment_half_width - 1.8, -590), backgroundcolor='white')
    plt.annotate('both ' + str(int((peak_right + peak_left) / 2)) + '±' + \
        str(int((reference_right + reference_left) / 2)) + ' ppm',\
        xy=(segment_half_width - 1.8, -650), backgroundcolor='white')
        
    # Axis labels and border settings
    plt.xlabel('Time [transit durations from mid-transit]')
    plt.ylabel('Relative flux [ppm]')
    plt.xticks(numpy.arange(-5, 6, 1))
    plt.xlim([-segment_half_width, segment_half_width])
    plt.ylim([-700, 500])
    
    # Save file
    #filename = str(segment_half_width) + '_' + str(polynom_degree) + '_' + \
    #    str(use_max_ttv) + '_' + str(use_ttv) + '_' + str(blinding * 2)
    #plt.savefig(path + filename + '.png', bbox_inches = 'tight')

    plt.show()


def main():
    # Parameters for this KOI analysis
    koi_name = '914.01'  # 711.02
    
    # If desired, We can clip high outlier values. See function for details.
    # High outliers are usually caused by cosmic ray hits.
    # After detrending, we can clip again more agressively.
    # Currently, no low outliers are clipped, as they occur rarely and are 
    # difficult to determine during transit (without transit model)
    # Therefore, check validity of result without any clipping!
    sigma_clip_before_detrending = 5
    sigma_clip_after_detrending = 3

    # LC data is 30 minutes, SC is 1 minute. If none are used, we fail :-)
    use_short_cadence = True
    use_long_cadence = True

    # 'pdcsap_flux' (with instrumental trends removed)
    # or 'sap_flux' (the simple aperture photometry)
    data_type = 'sap_flux'

    segment_half_width = 3  # transit durations on each side
    polynom_degree = 3  # 1=line, 2=parabola, etc.
    use_ttv = False  # 'False' uses the linear ephemeris
    use_max_ttv = 15  # minutes. Transits with higher TTVs are discarded
    blinding = 0.7  # transit durations on each side of mid-transit
    make_detrending_figures = False  # One PNG plot for each transit

    # For the weighted running mean, we define the window size
    window_size = 0.05  # as fraction of transit duration
    
    # To measure the height of the peak, we search between 'peak_window_start'
    # and mid-transit (on each side separately), in units of transit duration
    peak_window_start = 0.6

    # Start of main work
    print('Loading catalog...')
    masterdata = load_master_data_catalog(path + 'masterdata_3.csv')
    this_koi_phase, this_koi_flux, this_koi_weights = get_koi_phasefold(
        koi_name=koi_name,
        masterdata=masterdata,
        sigma_clip_before_detrending=sigma_clip_before_detrending,
        sigma_clip_after_detrending=sigma_clip_after_detrending,
        use_max_ttv=use_max_ttv,
        make_detrending_figures=make_detrending_figures,
        blinding=blinding,
        use_ttv=use_ttv,
        segment_half_width=segment_half_width,
        polynom_degree=polynom_degree,
        use_long_cadence=use_long_cadence)

    # Save phase-flux-weight data to file
    # print('Saving...')
    # numpy.savetxt(path + 'filename.csv', sorted_data, fmt='%1.8f')

    print('Creating weighted running mean (this can take several minutes)...')
    window_size_points = int((numpy.size(this_koi_flux) / (segment_half_width * 2) * window_size))
    print('Window size', window_size, 'TDs, ', window_size_points, 'data points')
    mid_window_times, mean_values = weighted_average(
        this_koi_phase,
        this_koi_flux,
        this_koi_weights,
        window_size_points)

    # Convert flux values to ppm (parts per million) with 0 as nominal flux
    million = 1000000
    mid_window_times = numpy.asarray(mid_window_times)
    mean_values = numpy.asarray(mean_values)
    mean_values = mean_values * million - million
    this_koi_flux = this_koi_flux * million - million

    # Get peaks and comparison levels of weighted running mean
    peak_left = get_max(mid_window_times, mean_values, start=-peak_window_start, end=0)
    peak_right = get_max(mid_window_times, mean_values, start=0, end=peak_window_start)
    reference_left = get_max(mid_window_times, mean_values, start=-segment_half_width, end=-peak_window_start)
    reference_right = get_max(mid_window_times, mean_values, start=peak_window_start, end=segment_half_width)
    print('left ' + str(int(peak_left)) + '±' + str(int(reference_left)) + ' ppm')
    print('right ' + str(int(peak_right)) + '±' + str(int(reference_right)) + ' ppm')
    print('both ' + str(int((peak_right + peak_left) / 2)) + '±' + \
        str(int((reference_right + reference_left) / 2)) + ' ppm')

    print('Making figure...')
    make_figure(mid_window_times, mean_values, mid_window_times, mean_values,
        window_size, segment_half_width, blinding, polynom_degree, use_max_ttv, use_ttv,
        peak_left, reference_left, peak_right, reference_right)


if __name__ == "__main__":
    path = 'I:/D/Research/phase_folder/Holczer/'
    main()
