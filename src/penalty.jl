slow_schedule(ρ, iteration) = iteration % 500 == 0 ? ρ*1.1 : ρ

fast_schedule(ρ, iteration) = iteration % 50 == 0 ? ρ*1.1 : ρ

function slow_schedule(T, n, ρ, iteration)
    if iteration % 500 == 0
        ρ_new = 1.1 * ρ
        diff = 1 / ρ_new - 1 / ρ
        for i in 1:n
            T[i,i] = T[i,i] + diff
        end
        ρ = ρ_new
    end

    return ρ
end

function fast_schedule(T, n, ρ, iteration)
    if iteration % 50 == 0
        ρ_new = 1.1 * ρ
        diff = 1 / ρ_new - 1 / ρ
        for i in 1:n
            T[i,i] = T[i,i] + diff
        end
        ρ = ρ_new
    end

    return ρ
end
