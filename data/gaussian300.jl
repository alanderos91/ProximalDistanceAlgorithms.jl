using ProximalDistanceAlgorithms, Random, StableRNGs, Distances, CSV, DataFrames

rng = StableRNG(5357)
σ = 0.2

# simulate three clusters
centroid = [[0.0, 0.0], [2.0, 2.0], [1.8, 0.5]]
nclass = [150, 50, 100]

# simulate three clusters
X1 = gaussian_cluster(centroid[1], nclass[1], rng=rng, sigma=σ)
X2 = gaussian_cluster(centroid[2], nclass[2], rng=rng, sigma=σ)
X3 = gaussian_cluster(centroid[3], nclass[3], rng=rng, sigma=σ)
X = [X1 X2 X3]

# class labels
classes = Int[]
for (k, n) in enumerate(nclass)
    for i in 1:n
        push!(classes, k)
    end
end

# save to file
feature1 = vec(X[1,:])
feature2 = vec(X[2,:])

df = DataFrame(
    feature1 = feature1,
    feature2 = feature2,
    classes  = classes,
)

CSV.write(joinpath("data", "gaussian300.dat"), df)
