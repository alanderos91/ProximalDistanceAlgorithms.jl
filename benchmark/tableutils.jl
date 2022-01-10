using Glob, CSV, DataFrames, Statistics, Latexify, Printf

function glob_benchmark_data(experiment;
    directory = "results",
    regex = r"(MMSubSpace{\d+}|ADMM|SteepestDescent|MM)(-CG|-LSQR|)")
    #
    # glob the data for specific experiment
    #
    pattern = experiment
    files = glob(pattern, directory)

    if isempty(files)
        error("""
            no files found matching pattern $(pattern) in $(directory)
        """)
    end

    #
    # assemble into a large DataFrame
    #
    df = DataFrame()
    for file in files
        #
        # match characters up to the second '_' or digit
        #
        m = match(regex, basename(file))
        algorithm = string(m.match)

        tmp = CSV.read(file, DataFrame)
        tmp[!, :algorithm] .= algorithm

        df = vcat(df, tmp)
    end

    return df
end

function glob_convergence_data(experiment;
    directory = "results",
    regex = r"(MMSubSpace{\d+}|ADMM|SteepestDescent|MM)(-CG|-LSQR|)")
    #
    # glob the data for specific experiment
    #
    pattern = experiment
    files = glob(pattern, directory)

    if isempty(files)
        error("""
            no files found matching pattern $(pattern) in $(directory)
        """)
    end

    #
    # assemble into a large DataFrame
    #
    df = DataFrame()
    for file in files
        m = match(regex, basename(file))
        algorithm = string(m.match)

        tmp = CSV.read(file, DataFrame)
        tmp[!, :algorithm] .= algorithm

        slice = DataFrame(tmp[end, :])
        select!(slice, Not([:outer, :inner]))
        
        df = vcat(df, slice)
    end

    return df
end

function algorithm_ordering(str)
    r1, r2 = if contains(str, "SDADMM")
        10, 0
    elseif contains(str, "MMSubSpace")
        subspace_size_search = match(r"\d+", str)
        subspace_size = parse(Int, subspace_size_search.match)
        4, subspace_size
    elseif contains(str, "ADMM")
        3, 0
    elseif contains(str, "SteepestDescent")
        2, 0
    elseif contains(str, "MM")
        1, 0
    else
        0, 0
    end

    r3 = if contains(str, "-CG")
        1
    elseif contains(str, "-LSQR")
        2
    else
        0
    end
    
    return (r1, r2, r3)
end

function print_tabular_header(left_columns, metric_columns, algorithms)
    number_left_cols = length(left_columns)
    number_metrics = length(metric_columns)
    number_algorithms = length(algorithms)

    # Begin a new tabular environment with the given column specification.
    spec1 = repeat("r", number_left_cols)
    spec2 = repeat(string("|", repeat("c", number_algorithms)), number_metrics)
    column_specification = string("{", spec1, spec2, "}")
    println("\\begin{tabular}$(column_specification)")

    # Add the table header with given metrics.
    header_left = repeat("& ", number_left_cols-1)
    header_metrics = ""
    for metric in metric_columns
        header_metrics = string(header_metrics, " & \\multicolumn{$(number_algorithms)}{c}{$(metric)}")
    end
    println("\\hline")
    println(header_left, header_metrics, "\\\\")
    println("\\hline")

    # Add the table subheader with given left columns and algorithms.
    subheader_left = join(left_columns, " & ")
    subheader_metrics = repeat(string(" & ", join(algorithms, " & ")), number_metrics)
    println(subheader_left, subheader_metrics, "\\\\")
    println("\\hline")

    return nothing
end

print_tabular_footer() = println("\\hline\n\\end{tabular}")

function print_latex_table(df; metric_columns=Symbol[], left_columns=Symbol[])
    algorithms = unique(df.algorithm)

    formatf = FancyNumberFormatter("%.3g", s"\g<mantissa> \\times 10^{\g<exp>}")
    formatter = function(x)
        str = formatf(x)
        str = replace(str, r"({-0+)" => "{-")
        return str
    end

    number_left_cols = length(left_columns)
    number_algorithms = length(algorithms)
    blanks = ["" for _ in 1:number_algorithms]
    
    print_tabular_header(left_columns, metric_columns, algorithms)

    for sub in groupby(df, left_columns)
        # Print extra problem information on the left columns.
        row1_left, row1_metrics = join(sub[1, left_columns], " & "), ""
        row2_left, row2_metrics = repeat("& ", number_left_cols-1), ""

        # For each metric...
        for metric in metric_columns
            # Select the target column and format the data.
            if metric == :time
                data_raw = sub[:, :time_avg]
                data_extra = sub[:, :time_std]
            elseif metric == :iterations
                data_raw = sub[:, :inner]
                data_extra = sub[:, :outer]
            elseif metric == :ARI || metric == :NMI
                data_raw = sub[:, Symbol(metric, :_avg)]
                data_extra = sub[:, Symbol(metric, :_std)]
            else
                data_raw = sub[:, metric]
                data_extra = []
            end

            for i in eachindex(data_raw)
                data_raw[i] = round(data_raw[i], sigdigits=3)
            end

            for i in eachindex(data_extra)
                data_extra[i] = round(data_extra[i], sigdigits=3)
            end

            data1_formatted = map(x -> string(formatter(x)), data_raw)
            data2_formatted = isempty(data_extra) ? blanks : map(x -> string("\$(", formatter(x), ")\$"), data_extra)

            # Highlight the optimal value to indicate the "best" methods.
            optimal_value = (metric == :ARI || metric == :NMI) ? maximum(data_raw) : minimum(data_raw)
            for i in eachindex(data1_formatted)
                if isapprox(data_raw[i], optimal_value, rtol=1e-4)
                    data1_formatted[i] = "\$\\bf{$(data1_formatted[i])}\$"
                else
                    data1_formatted[i] = "\$$(data1_formatted[i])\$"
                end
            end

            row1_metrics = string(row1_metrics, " & ", join(data1_formatted, " & "))
            row2_metrics = string(row2_metrics, " & ", join(data2_formatted, " & "))
        end
        println(row1_left, row1_metrics, "\\\\")
        println(row2_left, row2_metrics, "\\\\")
    end

    print_tabular_footer()

    return nothing
end

##############################
# Example: Metric Projection #
##############################

# Set target columns.
input_columns = [:nodes]
grouping_columns = [input_columns; :algorithm]

# Preprocess benchmark data.
df1 = glob_benchmark_data("metric*benchmark.dat")
df1 = combine(groupby(df1, grouping_columns), [
        :time => mean => :time_avg,
        :time => std => :time_std,
        :memory => mean => :memory_avg,
        :memory => std => :memory_std,
        :outer => first => :outer,
        :inner => first => :inner,
    ])

# Preprocess convergence data.
df2 = glob_convergence_data("metric*convergence.dat")

# Sort rows by problem size and algorithms.
df = leftjoin(df1, df2, on=grouping_columns)
sort!(df, [input_columns; order(:algorithm, by=algorithm_ordering)])

# Select algorithms that should appear in table.
filter!(row -> !contains(row.algorithm, "MMSubSpace"), df)
df[!, :algorithm] .= replace(x -> contains(x, "SteepestDescent") ? "SD" : x, df.algorithm)

# Scale select columns for readability.
df[!, :loss] .= df.loss * 1e-3
df[!, :distance] .= df.distance * 1e3

print_latex_table(df, metric_columns=[:time, :loss, :distance, :iterations], left_columns=input_columns)

##############################
# Example: Convex Regression #
##############################

# Set target columns.
input_columns = [:features, :samples]
grouping_columns = [input_columns; :algorithm]

# Preprocess benchmark data.
df1 = glob_benchmark_data("cvxreg*benchmark.dat")
df1 = combine(groupby(df1, grouping_columns), [
        :time => mean => :time_avg,
        :time => std => :time_std,
        :memory => mean => :memory_avg,
        :memory => std => :memory_std,
        :outer => first => :outer,
        :inner => first => :inner,
        :mse => first => :mse,
    ])

# Preprocess convergence data.
df2 = glob_convergence_data("cvxreg*convergence.dat")

# Sort rows by problem size and algorithms.
df = leftjoin(df1, df2, on=grouping_columns)
sort!(df, [input_columns; order(:algorithm, by=algorithm_ordering)])

# Select algorithms that should appear in table.
filter!(row -> !contains(row.algorithm, "MMSubSpace"), df)
filter!(row -> !contains(row.algorithm, "-LSQR"), df)
df[!, :algorithm] .= replace(x -> contains(x, "SteepestDescent") ? "SD" : x, df.algorithm)
df[!, :algorithm] .= replace(x -> contains(x, "-CG") ? first(split(x, '-')) : x, df.algorithm)

# Scale select columns for readability.
df[!, :loss] .= df.loss * 1e3
df[!, :distance] .= df.distance * 1e4
df[!, :mse] .= df.mse * 1e3

print_latex_table(df, metric_columns=[:time, :loss, :distance, :mse], left_columns=input_columns)

# Bonus: CG and LSQR comparisons with MM

df1 = glob_benchmark_data("cvxreg-d=20*MM*benchmark.dat")
df1 = combine(groupby(df1, grouping_columns), [
        :time => mean => :time_avg,
        :time => std => :time_std,
        :outer => first => :outer,
        :inner => first => :inner,
    ])

# Preprocess convergence data.
df2 = glob_convergence_data("cvxreg-d=20*MM*convergence.dat")

# Sort rows by problem size and algorithms.
df = leftjoin(df1, df2, on=grouping_columns)
sort!(df, [input_columns; order(:algorithm, by=algorithm_ordering)])

# Select algorithms that should appear in table.
filter!(row -> !contains(row.algorithm, "MMSubSpace"), df)
filter!(row -> !contains(row.algorithm, "SteepestDescent"), df)
filter!(row -> !contains(row.algorithm, "ADMM"), df)

df[!, :algorithm] .= replace(x -> contains(x, "MM-LSQR") ? "LSQR" : x, df.algorithm)
df[!, :algorithm] .= replace(x -> contains(x, "MM-CG") ? "CG" : x, df.algorithm)

# Scale select columns for readability.
df[!, :loss] .= df.loss * 1e6
df[!, :distance] .= df.distance * 1e6

print_latex_table(df, metric_columns=[:time, :loss, :distance, :iterations], left_columns=input_columns)

# Bonus: CG and LSQR comparisons with ADMM

df1 = glob_benchmark_data("cvxreg-d=20*ADMM*benchmark.dat")
df1 = combine(groupby(df1, grouping_columns), [
        :time => mean => :time_avg,
        :time => std => :time_std,
        :outer => first => :outer,
        :inner => first => :inner,
    ])

# Preprocess convergence data.
df2 = glob_convergence_data("cvxreg-d=20*ADMM*convergence.dat")

# Sort rows by problem size and algorithms.
df = leftjoin(df1, df2, on=grouping_columns)
sort!(df, [input_columns; order(:algorithm, by=algorithm_ordering)])

# Select algorithms that should appear in table.
filter!(row -> !contains(row.algorithm, "MMSubSpace"), df)
filter!(row -> !contains(row.algorithm, "SteepestDescent"), df)
# filter!(row -> !contains(row.algorithm, "MM"), df)

df[!, :algorithm] .= replace(x -> contains(x, "ADMM-LSQR") ? "LSQR" : x, df.algorithm)
df[!, :algorithm] .= replace(x -> contains(x, "ADMM-CG") ? "CG" : x, df.algorithm)

# Scale select columns for readability.
df[!, :loss] .= df.loss * 1e6
df[!, :distance] .= df.distance * 1e6

print_latex_table(df, metric_columns=[:time, :loss, :distance, :iterations], left_columns=input_columns)

##############################
# Example: Convex Clustering #
##############################

# Set target columns.
input_columns = [:dataset, :features, :samples, :classes]
grouping_columns = [input_columns; :algorithm]

# Preprocess benchmark data.
df1 = glob_benchmark_data("cvxclst*benchmark.dat")
df1 = combine(groupby(df1, grouping_columns), [
        :time => mean => :time_avg,
        :time => std => :time_std,
        :memory => mean => :memory_avg,
        :memory => std => :memory_std,
        :outer => first => :outer,
        :inner => first => :inner,
        :ARI => mean => :ARI_avg,
        :ARI => std => :ARI_std,
        :NMI => mean => :NMI_avg,
        :NMI => std => :NMI_std,
    ])

# Preprocess convergence data.
df2 = glob_convergence_data("cvxclst*convergence.dat")

# Sort rows by problem size and algorithms.
df = leftjoin(df1, df2, on=grouping_columns)
sort!(df, [:samples; order(:algorithm, by=algorithm_ordering)])

# Select algorithms that should appear in table.
filter!(row -> !contains(row.algorithm, "MMSubSpace"), df)
filter!(row -> !contains(row.algorithm, "-CG"), df)
df[!, :algorithm] .= replace(x -> contains(x, "SteepestDescent") ? "SD" : x, df.algorithm)
df[!, :algorithm] .= replace(x -> contains(x, "-LSQR") ? first(split(x, '-')) : x, df.algorithm)

# Scale select columns for readability.
df[!, :distance] .= df.distance * 1e5

print_latex_table(df, metric_columns=[:time, :loss, :distance, :ARI, :NMI], left_columns=input_columns)

##############################
# Example: Image Denoising   #
##############################

# Set target columns.
input_columns = [:image, :width, :height]
grouping_columns = [input_columns; :algorithm]

# Preprocess benchmark data.
df1 = glob_benchmark_data("imgtvd*benchmark.dat")
df1 = combine(groupby(df1, grouping_columns), [
        :time => mean => :time_avg,
        :time => std => :time_std,
        :memory => mean => :memory_avg,
        :memory => std => :memory_std,
        :outer => first => :outer,
        :inner => first => :inner,
        :s => first => :s,
        :MSE => first => :MSE,
        :PSNR => first => :PSNR,
    ])

# Preprocess convergence data.
df2 = glob_convergence_data("imgtvd*convergence.dat")

# Sort rows by problem size and algorithms.
df = leftjoin(df1, df2, on=grouping_columns)
sort!(df, [input_columns; order(:algorithm, by=algorithm_ordering)])

# Select algorithms that should appear in table.
filter!(row -> !contains(row.algorithm, "MMSubSpace"), df)
filter!(row -> !contains(row.algorithm, "-CG"), df)
df[!, :algorithm] .= replace(x -> contains(x, "SteepestDescent") ? "SD" : x, df.algorithm)
df[!, :algorithm] .= replace(x -> contains(x, "-LSQR") ? first(split(x, '-')) : x, df.algorithm)

# Scale select columns for readability.
df[!, :distance] .= df.distance * 1e3
df[!, :MSE] .= df.MSE * 1e5

print_latex_table(df, metric_columns=[:time, :loss, :distance, :MSE, :PSNR], left_columns=input_columns)

##############################
# Example: Condition Number  #
##############################

# Set target columns.
input_columns = [:p, :a]
grouping_columns = [input_columns; :algorithm]

# Preprocess benchmark data.
df1 = glob_benchmark_data("condnum*benchmark.dat")
df1 = combine(groupby(df1, grouping_columns), [
        :time => mean => :time_avg,
        :time => std => :time_std,
        :memory => mean => :memory_avg,
        :memory => std => :memory_std,
        :outer => first => :outer,
        :inner => first => :inner,
        :condM => first => :condM,
        :condX => first => :condX,
        :fidelity => first => :fidelity,
    ])

# Preprocess convergence data.
df2 = glob_convergence_data("condnum*convergence.dat")

# Sort rows by problem size and algorithms.
df = leftjoin(df1, df2, on=grouping_columns)
sort!(df, [input_columns; order(:algorithm, by=algorithm_ordering)])

# Select algorithms that should appear in table.
filter!(row -> !contains(row.algorithm, "MMSubSpace"), df)
df[!, :algorithm] .= replace(x -> contains(x, "SteepestDescent") ? "SD" : x, df.algorithm)

# Scale select columns for readability.
df[!, :condM] .= round.(df.condM, sigdigits=3)
df[!, :time_avg] .= df.time_avg * 1e3
df[!, :time_std] .= df.time_std * 1e3
df[!, :loss] .= df.loss * 1e3
df[!, :distance] .= df.distance * 1e5

print_latex_table(df, metric_columns=[:time, :loss, :distance, :condX], left_columns=[:p, :condM, :a])
