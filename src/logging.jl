# convergence metrics for MM algorithms

struct MMLogger{vecT}
    g::vecT     # norm of gradient
    loss::vecT
    objective::vecT
    penalty::vecT
    iteration::Vector{Int}
    sample_rate::Int
end

function MMLogger(hint::Integer, sample_rate = 50)
    g = sizehint!(Float64[], hint)
    loss = sizehint!(Float64[], hint)
    objective = sizehint!(Float64[], hint)
    penalty = sizehint!(Float64[], hint)
    iteration = sizehint!(Int[], hint)

    return MMLogger(g, loss, objective, penalty, iteration, sample_rate)
end

function (logger::MMLogger)(data, iteration)
    # log history every 50 iterations
    if iteration % logger.sample_rate == 0
        # retrieve info stored in data
        g, loss, objective, penalty = data

        # update fields
        push!(logger.g, g)
        push!(logger.loss, loss)
        push!(logger.objective, objective)
        push!(logger.penalty, penalty)
        push!(logger.iteration, iteration)
    end

    return nothing
end

@recipe function plot(logger::MMLogger)
    title --> "MM convergence history"
    legend --> :bottom
    xlabel --> "iteration"
    xscale --> :log10
    yscale --> :log10

    # loss
    @series begin
        seriestype := :scatter
        label --> "loss"
        logger.iteration, logger.loss
    end

    # objective
    @series begin
        seriestype := :scatter
        label --> "objective"
        logger.iteration, logger.objective
    end

    # penalty
    @series begin
        seriestype := :scatter
        label --> "penalty"
        logger.iteration, logger.penalty
    end

    # norm of gradient
    @series begin
        seriestype := :scatter
        label --> "gradient"
        logger.iteration, logger.g
    end
end

# convergence metrics for SteepestDescent algorithms

struct SDLogger{vecT}
    γ::vecT     # step size
    g::vecT     # norm of gradient
    loss::vecT
    objective::vecT
    penalty::vecT
    iteration::Vector{Int}
    sample_rate::Int
end

function SDLogger(hint::Integer, sample_rate = 50)
    γ = sizehint!(Float64[], hint)
    g = sizehint!(Float64[], hint)
    loss = sizehint!(Float64[], hint)
    objective = sizehint!(Float64[], hint)
    penalty = sizehint!(Float64[], hint)
    iteration = sizehint!(Int[], hint)

    return SDLogger(γ, g, loss, objective, penalty, iteration, sample_rate)
end

function (logger::SDLogger)(data, iteration)
    # log history every 50 iterations
    if iteration % logger.sample_rate == 0
        # retrieve info stored in data
        γ, g, loss, objective, penalty = data

        # update fields
        push!(logger.γ, γ)
        push!(logger.g, g)
        push!(logger.loss, loss)
        push!(logger.objective, objective)
        push!(logger.penalty, penalty)
        push!(logger.iteration, iteration)
    end

    return nothing
end

@recipe function plot(logger::SDLogger)
    title --> "SD convergence history"
    legend --> :bottom
    xlabel --> "iteration"
    xscale --> :log10
    yscale --> :log10

    # loss
    @series begin
        seriestype := :scatter
        label --> "loss"
        logger.iteration, logger.loss
    end

    # objective
    @series begin
        seriestype := :scatter
        label --> "objective"
        logger.iteration, logger.objective
    end

    # penalty
    @series begin
        seriestype := :scatter
        label --> "penalty"
        logger.iteration, logger.penalty
    end

    # norm of gradient
    @series begin
        seriestype := :scatter
        label --> "gradient"
        logger.iteration, logger.g
    end
end
