'''
This script will calculate and plot the number of Monthly Active Users stored in MongoDB collection

This script assumes that the collection is stored by years

Currently, we hardcode the years to 2016, 2017, 2018 which fits the scope of the project

Requirements:
- source activate HackerNews
- import the data into MongoDB `mongoimport --db HackerNews --collections hn_{{ $year }} {{ $year }}.fmt   
'''


def connect_to_db():
    '''
    Connects to MongoDB using pymongo and return a zipped collections
    '''
    from pymongo import MongoClient
    client = MongoClient()
    db = client.HackerNews

    hn_2018 = db.hn_2018
    hn_2017 = db.hn_2017
    hn_2016 = db.hn_2016

    collections = [hn_2016, hn_2017, hn_2018]

    return collections


def get_MAUS(years, months, collections, save=True):
    import datetime
    from datetime import date

    epoch_dt = date(1970, 1, 1)

    maus = []
    tic = time.time()

    for year_count in range(len(years)):
        print("=== Year {} ===".format(years[year_count]))
        mau = []
        for month_count in range(len(months)):
            cursor = collections[year_count].find()
            year = years[year_count]
            if month_count == 11:  # To settle the month of Dec

                start_time = int(
                    (date(int(year), 12, 1) - epoch_dt).total_seconds())
                end_time = int(
                    (date(int(year) + 1, 1, 1) - epoch_dt).total_seconds())
                temp = []
                for j in cursor:
                    if int(j['time']) > start_time and int(j['time']) < end_time:
                        temp.append(j)
                usr_profiles = set([i['by'] for i in temp])
                mau.append(len(usr_profiles))
                print("{} {} MAU: {} time elapsed: {} secs".format(months[month_count], date(
                    int(year)+1, 12, 1), len(usr_profiles), time.time() - tic))
            else:
                start_time = int(
                    ((date(int(year), month_count + 1, 1)) - epoch_dt).total_seconds())
                end_time = int(
                    ((date(int(year), month_count + 2, 1)) - epoch_dt).total_seconds())
                temp = []
                for j in cursor:
                    if int(j['time']) > start_time and int(j['time']) < end_time:
                        temp.append(j)
                usr_profiles = set([i['by'] for i in temp])
                mau.append(len(usr_profiles))
                print("{} {} MAU: {} time elapsed: {} secs".format(months[month_count], date(
                    int(year), month_count + 1, 1), len(usr_profiles), time.time() - tic))
    if save:

        file_name = 'MAUS.text'
        with open(file_name, 'w') as f:
            for i in maus:
                f.write(i)
            f.close()
    return maus


if __name__ == '__main__':
    from bokeh.io import show
    from bokeh.io import output_file
    from bokeh.plotting import figure

    months = ['jan', 'feb', 'mar', 'apr', 'may', 'jun',
              'jul', 'aug', 'sep', 'oct', 'nov', 'dec']
    years = ['2016', '2017', '2018']

    collections = connect_to_db()
    maus = get_MAUS(years, months, collections)

    for i in range(len(maus)):
        year = years[i]
        count = maus[i]
        output_file('mau_{}.html'.format(year))
        p = figure(x_range=months, plot_height=400, plot_width=600,
                   title="Monthly Active Users for Year {}".format(year))
        p.xaxis.major_label_text_font_size = "10pt"
        p.vbar(x=months, top=count, width=0.3)

        p.xgrid.grid_line_color = None
        p.y_range.start = 0

        show(p)
