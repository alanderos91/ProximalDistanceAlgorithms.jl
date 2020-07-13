using Random, Distances, CSV, DataFrames

# https://smorbieu.gitlab.io/generate-datasets-to-understand-some-clustering-algorithms-behavior/

function generate_spiral_data(n1, n2;
    max_radius = 7.0,
    x_shift = 2.5,
    y_shift = 2.5,
    angle_start = 2.5 * π,
    noise_variance = 0.1,)
    #

    # first spiral
    angle1  = [angle_start + angle_start * (i / n1) for i in 0:n1-1]
    radius1 = [max_radius * (n1 + n1 / 5 - i) / (n1 + n1 / 5) for i in 0:n1-1]
    x1      = radius1 .* sin.(angle1)
    y1      = radius1 .* cos.(angle1)

    # second spiral
    angle2  = [angle_start + angle_start * (i / n2) for i in 0:n2-1]
    radius2 = [max_radius * (n2 + n2 / 5 - i) / (n2 + n2 / 5) for i in 0:n2-1]
    x2      = -radius2 .* sin.(angle2)
    y2      = -radius2 .* cos.(angle2)

    # combine, add noise, and shift
    x = [x1; x2]
    y = [y1; y2]

    z1 = randn(n1 + n2)
    z2 = randn(n1 + n2)

    x .+= x_shift .+ noise_variance .* z1
    y .+= y_shift .+ noise_variance .* z2

    # class labels
    nclass  = [n1, n2]
    classes = Int[]
    for (k, n) in enumerate(nclass)
        for i in 1:n
            push!(classes, k)
        end
    end

    return x, y, classes
end

n1 = parse(Int, ARGS[1])
n2 = parse(Int, ARGS[2])

Random.seed!(5357)

# simulate
feature1, feature2, classes = generate_spiral_data(n1, n2, noise_variance = 0.3)

# save to file
df = DataFrame(
    feature1 = feature1,
    feature2 = feature2,
    classes  = classes,
)

CSV.write("spiral$(n1+n2).dat", df)
